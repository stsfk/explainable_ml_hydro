# Libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, lubridate, RcppRoll, SHAPforxgboost, zeallot, xgboost, viridis, hrbrthemes, cowplot)

# Prepare -----------------------------------------------------------------
# Generate feature engineering hyperparameters in in "modeling.R".

SEED <- 88192
set.seed(SEED)
start_time <- Sys.time()

nthread <- parallel::detectCores() - 2

data_raw <- read_csv(
  "./data/rainfall_runoff.data",
  col_types = cols(
    datetime = col_datetime(format = ""),
    X = col_double(),
    Y = col_double()
  )
)

data <- data_raw

feature_vector <- function(x, s, e){
  # This is a function to return vector of rainfall depth feature,
  # s is the start location
  # e is the end location
  # rainfall depth between t-e and t-s is computed
  
  e <- e + 1
  
  if (s == 0){
    out <- roll_sum(x, e, align = "right", fill = NA)
  } else {
    out <- roll_sum(x, e, align = "right", fill = NA) - roll_sum(x, s, align = "right", fill = NA)
  }
  
  out
}

gen_s_e <- function(m = 500, q = 1.5, l = 6, option){
  # This function to generate s and e pairs for computing the rainfall depth features
  # based on the feature engineering hyperparameters m, q, and l
  
  # Detailed rain ts to be included
  s_e_l <- tibble(s = c(0:l), 
                  e = c(0:l))
  
  # Aggregated rain time depth feature
  lis <- list()
  N <- 1
  lis[[N]] <- 2
  sn <- lis[[N]]
  
  while(sn < m - l){
    N <- N + 1
    lis[[N]] <- ceiling(2*(q^(N-1)))
    sn <- sn + lis[[N]]
  }
  lis[N] <- NULL
  
  if ((m - l - sum(unlist(lis))) >= lis[[length(lis)]]){ 
    # (m - l - sum(unlist(lis))) is length of remained time steps
    # lis[[length(lis)]] the length of the last step
    lis[[N]] <- m - l - sum(unlist(lis))
  } else {
    # add remained steps to current one
    lis[[length(lis)]] <- lis[[length(lis)]] + m - l - sum(unlist(lis))
  }
  
  s <- (unlist(lis) %>% cumsum()) + l # index from the beginning
  e <- s
  s <- c(l, s[-length(s)]) + 1
  
  s_e <- tibble(s = s,
                e = e)
  
  if (option == 1){
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  }
  
  if (option == 2){
    s_e <- rbind(s_e_l, s_e)
    s_e[,1] <- 0 # key line
    return(s_e)
  } 
  
  if (option == 3){
    s_e[,1] <- 0 # key line
    
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  }
  
  if (option == 4){
    s_e[,1] <- s_e[[1,1]] # key line
    
    s_e <- rbind(s_e_l, s_e)
    return(s_e)
  } 
}

gen_feature_para <- function(N_FEATURE_SETS, UB, LB){
  # This function generates N_FEATURE_SETS sets of of feature engineering hyperparameters
  # UB and LB are lower and upper bounds of the random values
  
  ms <- sample(x = LB[1]:UB[1], size = N_FEATURE_SETS, replace = T)
  qs <- runif(n = N_FEATURE_SETS, min = LB[2], max = UB[2])
  ls <- sample(x = LB[3]:UB[3], size = N_FEATURE_SETS, replace = T)
  
  out <- tibble(m = ms,
                q = qs,
                l = ls)
  
  out
}

gen_feature <- function(para, option = 1){
  # This is a function for creating features
  #   
  #   y: the output variable
  #   x: the original predictor
  #   s_e: the index of new predictors
  #   keep_colume: the name of additional columns to be kept, such as wet dry condition, time index, and id of event
  
  c(m, q, l) %<-% para
  
  s_e <- gen_s_e(m = m, q = q, l = l, option = option)
  
  # create a vector of the original predictor
  org_pred <- data$X
  
  # create out tibble
  out <- data %>%
    select(Y)
  
  # create features and name the columns based on the "s" and "e" index
  for (i in 1:nrow(s_e)){
    c(s, e) %<-% s_e[i,]
    
    var_name <- paste0("X", s, "_", e)
    out[var_name] <- feature_vector(org_pred, s, e)
  }
  
  out
}

N_PARAS <- 10 # 10 experiment
LB <- c(12, 1.1, 1) # experiment
UB <- c(500, 5, 72) # experiment

# Generate N_PARAS set of feature engineering hyperparameters
paras <- gen_feature_para(N_PARAS * 2, UB, LB)
paras <- paras %>%
  filter((m - 2) > l) %>%
  dplyr::slice(c(1:N_PARAS))

# Split the data
train_and_validation_fold <- data %>% 
  mutate(ind =(!is.na(Y) & datetime < ymd("2007-01-20"))) %>%
  pull(ind) %>%
  which() #  2735 records

validation_fold <- data %>% 
  mutate(ind = datetime >= ymd("2007-01-15") & datetime <= ymd("2007-01-20")) %>%
  pull(ind) %>%
  which()
validation_fold <- (train_and_validation_fold %in% validation_fold) %>%
  which() # 721 records
train_inner_fold <- setdiff(seq_along(train_and_validation_fold), validation_fold)

test_fold <- data %>% 
  mutate(ind =(!is.na(Y))) %>%
  pull(ind) %>%
  which()
test_fold <- setdiff(test_fold, train_and_validation_fold) # 2264 records

full_event_divide_df <- tibble(
  train_and_validation_fold =  list(train_and_validation_fold),
  validation_fold = list(validation_fold),
  test_fold = list(test_fold)
)

# Catchment response time -------------------------------------------------

# Obtain shap values for model i
compute_shap <- function(option, i){
  # This function compute shap value for model_i_j, 
  #   feature engineering hyperpara set i and outer CV iter# j
  
  model <- xgb.load(paste0("./data/results/exp", option, "/xgb_", i, ".model"))
  
  data_feature <- gen_feature(paras[i,], option = option)
  data_feature[is.na(data_feature)] <- 0
  
  shap_values <- shap.values(xgb_model = model, X_train = data_feature[,-1])
  
  shap_values
}

distribute_shap <- function(shap_values){
  # This function compute the average shap contribution of rainfalls at different time steps
  # This may also be intepreted as rainfalls to contribution to runoffs at different time steps ahead

  mean_shap_score <- shap_values$mean_shap_score
  
  feature_names <- mean_shap_score %>% names()
  
  rain_depth_feature_ind <- names(mean_shap_score) %>% str_detect("^X") %>% which()
  other_score <- mean_shap_score[-rain_depth_feature_ind]
  
  rain_depth_feature <- mean_shap_score[rain_depth_feature_ind]
  
  rain_depth_feature_df <- tibble(
    var_name = rain_depth_feature %>% names(),
    shap = rain_depth_feature
  )
  
  out <- gen_s_e(paras[i, ]$m, paras[i, ]$q, paras[i, ]$l, option) %>%
    mutate(var_name = paste0("X", s, "_", e)) %>%
    left_join(rain_depth_feature_df, by = "var_name") %>%
    mutate(shap_mean = shap/(e - s + 1))
  
  data_plot <- tibble(ind = 0:max(out$e),
                      s = cut(ind, c(-1, out$e), right = T, labels = (out$s))) %>%
    mutate(s = as.numeric(as.character(s))) %>%
    left_join(out, by = "s")
  
  data_plot
}

data_plot <- compute_shap(option = 1, i = 7) %>%
  distribute_shap()

ggplot(data_plot, aes(ind, shap_mean)) +
  geom_line(color = "red")  +
  scale_x_continuous(trans=scales::pseudo_log_trans(base = 10), breaks = c(0,1,5,10,50,100,500,1000),
                     labels = c(0,1," ",10," ",100, " ", 1000))+
  labs(x = "Time steps ahead of a rainfall record",
       y = "SHAP value of a rainfall record (impact on flow rate [CFS])") +
  theme_bw()
  
# Hydrograph separation ---------------------------------------------------

# The functions for compute SHAP values are included in the preceding code chunks.

prepare_plot <- function(shap_values){
  # this function prepares data_plot for showing the composition of hydrographs.

  data_plot <- shap_values$shap_score %>%
    as_tibble() %>%
    mutate(bias = shap_values$BIAS0 %>% unlist()) %>%
    dplyr::select(bias, everything()) %>%
    mutate(ind = 1:n()) %>%
    gather(item, value,-ind) %>%
    mutate(item = factor(item, levels = c(
      "bias", colnames(shap_values$shap_score)
    ) %>% rev()))
  
  
  data_plot$positive <- ifelse(data_plot$value >= 0, data_plot$value, 0)
  data_plot$negative <- ifelse(data_plot$value < 0, data_plot$value, -1e-36)
  
  old_y_text <- levels(data_plot$item)
  new_y_text <- str_replace(old_y_text, "_", ",t-") %>%
    str_replace("X", "D[t-")
  for (i in seq_along(new_y_text)){
    if (str_detect(new_y_text[i], "D")){
      new_y_text[i] <- str_c(new_y_text[i], "]")
      new_y_text[i] <- str_replace(new_y_text[i], ",", "*','*")
    }
  }
  data_plot$item <- factor(data_plot$item,
                           levels = old_y_text,
                           labels = new_y_text)
  
  data_plot
}

data_plot <- compute_shap(option = 1, i = 7) %>%
  prepare_plot()

data_plot_sub <- data_plot %>%
  filter(ind >2500, ind < 3000)

ggplot(data_plot_sub) + 
  geom_area(aes(ind, positive, fill = item), alpha=0.6 , size=0.1, colour="white")+
  geom_area(aes(ind, negative, fill = item), alpha=0.6 , size=0.1, colour="white")+
  scale_fill_viridis(discrete = T, labels = function(l) parse(text=l),   
                     guide = guide_legend(
                       direction = "horizontal",
                       title.position = "top",
                       label.position = "right",
                       label.hjust = 0,
                       label.vjust = 1,
                       ncol = 1
                     )) +
  labs(x = "Time step since the beginning of rainfall event",
       y = "Flow rate contribution [CFS]",
       fill = "Contributing rainfalls\nand other factors") +
  theme_ipsum(
    axis_title_just = "m",
    axis_title_size = 9,
    base_size = 8,
    strip_text_size = 8,
    plot_margin = margin(10, 10, 10, 10)
  ) +
  theme(legend.position = "right", 
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text=element_text(size=rel(1.1), face = "italic"),
        panel.spacing = unit(1, "lines"))


# Continuing impact of a rainfall depth record ----------------------------

shap_values <- compute_shap(option = 1, i = 5)

compute_shap_contribution <- function(shap_values, i= 5, ind = 542){
  # This function compute the continuing impact of a rainfall at ind time step
  # to runoffs at subsequent time steps.
  
  s_e <- gen_s_e(m = paras$m[i], q = paras$q[i], l = paras$l[i], option = 1)
  max_effect <- s_e$e %>% max()
  
  shap_scalar <- vector("double", max_effect + 1)
  contributed_shap <- vector("double", max_effect + 1)
  rain_depth <- data_feature$X0_0[ind]
  shap_score <- shap_values$shap_score %>%
    as_tibble()
  
  correspond_feature_indexs <- cut(0:max_effect, breaks = c(-1, s_e$e), right = T, labels = (1:nrow(s_e))) %>%
    as.character() %>% 
    as.numeric()
  
  for (i in seq_along(shap_scalar)){
    
    row_ind <- ind + i - 1
    col_ind <- correspond_feature_indexs[i] + 1
    
    correspond_feature <- data_feature[[row_ind, col_ind]]
    
    shap_scalar[[i]] <- ifelse(rain_depth/correspond_feature==Inf, 0, rain_depth/correspond_feature)
    contributed_shap[[i]] <- shap_scalar[[i]] * shap_score[[row_ind, col_ind - 1]]
  }
  
  tibble(
    ind = ind:(ind + max_effect), 
    contributed_shap = contributed_shap
  )
}

prepare_contribution_plot <- function(shap_values, shap_contribution){
  
  predictions <- tibble(
    pred = (shap_values$shap_score %>% rowSums()) + (shap_values$BIAS0 %>% unlist())
  ) %>%
    mutate(
      ind = 1:n()
    )
  
  data_plot <- predictions %>% 
    left_join(shap_contribution, by = "ind") %>%
    mutate(contributed_shap = replace(contributed_shap, is.na(contributed_shap), 0)) %>%
    mutate(other_contribution = pred - contributed_shap) %>%
    select(-pred) %>% 
    gather(item, value, contributed_shap, other_contribution) %>%
    mutate(item = factor(item, levels = c("other_contribution", "contributed_shap"),
                         labels = c("Other factors", "Contribution of peak rainfall")))
  
  data_plot$positive <- ifelse(data_plot$value >= 0, data_plot$value, 0)
  data_plot$negative <- ifelse(data_plot$value < 0, data_plot$value, -1e-36)
  
  data_plot
}

shap_contribution <- compute_shap_contribution(shap_values)
data_plot <- prepare_contribution_plot(shap_values, shap_contribution)

data_plot_sub <- data_plot %>%
  filter(ind > 500,
         ind < 700)

data_rain_sub <- data %>%
  mutate(ind = 1:n()) %>%
  filter(ind > 500,
         ind < 700 )

data_rain_sub2 <- data %>%
  mutate(ind = 1:n()) %>%
  filter(ind == 542)

a <- ggplot(data_plot_sub) +
  geom_area(
    aes(ind, positive, fill = item),
    alpha = 0.6 ,
    size = 0.1,
    colour = "white"
  ) +
  geom_area(
    aes(ind, negative, fill = item),
    alpha = 0.6 ,
    size = 0.1,
    colour = "white"
  ) +
  scale_fill_viridis(
    discrete = T,
    guide = guide_legend(
      title = "Contributing rainfalls\nand other factors",
      direction = "horizontal",
      title.position = "top",
      label.position = "right",
      label.hjust = 0,
      label.vjust = 0.5,
      ncol = 1
    )
  )  +
  labs(y =  "Flow rate contribution [CFS]",
       x = "Time step") +
  theme_bw()+
  theme(legend.justification = c(1,1),
        legend.position = c(0.99, 0.99))

b <- ggplot(data_rain_sub, aes(ind, X)) +
  geom_bar(stat = "identity") +
  geom_point(data = data_rain_sub2, color = "red") +
  annotate("text", x = 550, y = 1.4, label = "Peak rainfall", hjust = 0, color = "red") +
  labs(y =  "Rainfall intensity [inch/hour]",
       x = "Time step") +
  theme_bw()+
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank())

plot_grid(b, a, ncol = 1)
