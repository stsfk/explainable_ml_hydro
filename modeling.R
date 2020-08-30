# Libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, lubridate, RcppRoll, zeallot, xgboost)

# Prepare -----------------------------------------------------------------

SEED <- 88192
set.seed(SEED)
start_time <- Sys.time()

nthread <- parallel::detectCores() - 2

# Data --------------------------------------------------------------------

data_raw <- read_csv(
  "./data/rainfall_runoff.data",
  col_types = cols(
    datetime = col_datetime(format = ""),
    X = col_double(),
    Y = col_double()
  )
)

data <- data_raw

# Functions ---------------------------------------------------------------

# gen_feature_para generates random hyperparas `paras`
# gen_feature creates `data_feature`
# extract_data_by_event_id extract the data by `event_id_news`

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


# Pro-process -------------------------------------------------------------

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

# Create full_event_divide_df storing event index for training, validation and testing
full_event_divide_df <- tibble(
  train_and_validation_fold =  list(train_and_validation_fold),
  validation_fold = list(validation_fold),
  test_fold = list(test_fold)
)

# Modeling ----------------------------------------------------------------
tune_grid <- expand.grid(
  eta = c(0.02, 0.05, 0.1),
  max_depth = c(4, 6),
  colsample_bytree = c(0.7, 1),
  subsample = c(0.7, 1)
)

for (OPTION in c(1:4)){
  
  overall_results <- vector("list", N_PARAS)
  for (i in 1:N_PARAS){
    
    msg <- paste("Para set", i, "starts \n", sep = " ")
    cat(msg)
    
    para <- paras[i,]
    data_feature <- gen_feature(para, option = OPTION)
    data_feature[is.na(data_feature)] <- 0 # fill NA with zero, as there were no rain before the they study period
    preprocess_cols <- rep(1, ncol(data_feature) - 1)
    
    dtrain <- xgb.DMatrix(data = data.matrix(data_feature[train_and_validation_fold,-1]), 
                          label=data_feature$Y[train_and_validation_fold])
    
    dtrain_inner <- slice(dtrain, train_inner_fold)
    dval <- slice(dtrain, validation_fold)
    
    dtest <- xgb.DMatrix(data = data.matrix(data_feature[test_fold,-1]), label=data_feature$Y[test_fold])
    
    # Tuning hyperparameters
    tune_results <- vector("list", nrow(tune_grid))
    for (k in 1:nrow(tune_grid)){
      c(eta, max_depth, colsample_bytree, subsample) %<-% tune_grid[k,]
      
      # Validation
      watchlist <- list(val = dval)
      
      bst.cv <- xgb.train(data = dtrain_inner, tree_method = "hist",
                          objective = "reg:squarederror",
                          watchlist = watchlist,
                          nthread = nthread, max_bin = 256,
                          monotone_constraints = preprocess_cols,
                          eta = eta, max_depth = max_depth, colsample_bytree = colsample_bytree, subsample = subsample, nrounds = 2500,
                          gamma = 0, min_child_weight = 1, 
                          early_stopping_rounds = 10,
                          verbose = 0)
      
      tune_result <- bst.cv$evaluation_log[bst.cv$best_iteration,] %>% 
        unlist() %>% 
        matrix(nrow = 1) %>%
        data.frame()
      names(tune_result) <- bst.cv$evaluation_log[bst.cv$best_iteration,] %>% names()
      tune_results[[k]]  <- tune_result
      
      rm(bst.cv)
      gc()
    }
    
    tune_results <- tune_results %>% bind_rows()
    
    # Identify best hyperparameter set
    bst_k <- which.min(tune_results[['val_rmse']])
    c(eta, max_depth, colsample_bytree, subsample) %<-% tune_grid[bst_k,]
    nrounds <- tune_results[bst_k, "iter"]
    
    # Train tuned model and test
    watchlist <- list(test = dtest)
    bst <- xgb.train(data = dtrain, tree_method = "hist",
                     objective = "reg:squarederror",
                     watchlist = watchlist,
                     nthread = nthread, max_bin = 256,
                     monotone_constraints = preprocess_cols,
                     eta = eta, max_depth = max_depth, colsample_bytree = colsample_bytree, subsample = subsample, nrounds = nrounds,
                     gamma = 0, min_child_weight = 1, 
                     verbose = 0)
    
    # save model trained on entrie dtrain
    paths <- paste0("./data/results/exp", OPTION)
    dir.create(paths, showWarnings = FALSE)
    
    model_name <- paste0(paths, "/xgb_" , i, ".model")
    xgb.save(bst, model_name)
    
    # prediction df on the test fold
    pred_df <- tibble(datetime = data$datetime[test_fold],
                      ob = getinfo(dtest, "label"),
                      pred = predict(bst, dtest))
    
    # optimal hyperparameters
    optm_para <- tune_grid[bst_k,] %>% 
      as.data.frame() %>% 
      tibble() %>%
      mutate(nrounds = nrounds)
    
    # inner error for each set of hyperpara
    tune_results <- tune_results %>% 
      as.data.frame() %>% 
      tibble() %>%
      mutate(best = (bst_k == 1:n()))
    
    # save results: 
    # 1. pred_df, pred and ob on the test fold
    # 2. optm_para, optimal hyperparamter sets,
    # 3. tune_results, summary of the train and validation loss mean and std
    pred_name <- paste0(paths, "/pred_",i,".Rda")
    save(pred_df, optm_para, tune_results, file = pred_name)
    
    # evaluate external error
    val_rmse <- tune_results %>% dplyr::filter(best) %>% pull(val_rmse)
    
    # save all results for i
    overall_results[[i]] <- bst$evaluation_log[bst$niter,] %>% 
      data.frame() %>%
      tibble() %>%
      mutate(i = i, 
             pred = list(pred_df),
             tune_results = list(tune_results),
             val_rmse = val_rmse,
             optm_para = list(optm_para)) %>%
      select(i, everything())
    
    rm(bst)
    gc()
    
    cat("\n")
  }
  
  overall_results <- overall_results %>% 
    bind_rows()
  
  fname <- paste0(paths, "/overall_results.Rda")
  save(overall_results, paras, full_event_divide_df, file = fname)
}

end_time <- Sys.time()
end_time - start_time




