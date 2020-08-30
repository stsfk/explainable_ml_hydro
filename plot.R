# Libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, lubridate, RcppRoll, zeallot, xgboost)


# Data --------------------------------------------------------------------

# observation data

data_raw <- read_csv(
  "./data/rainfall_runoff.data",
  col_types = cols(
    datetime = col_datetime(format = ""),
    X = col_double(),
    Y = col_double()
  )
)

data <- data_raw
option <- 1

fname <- paste0("./data/results/exp", option, "/overall_results.Rda")
load(fname)


# Validation error vs test error ------------------------------------------

ggplot(overall_results, aes(val_rmse, test_rmse)) + 
  geom_point() +
  labs(x = "Validation RMSE [CFS]",
       y = "Test RMSE [CFS]")

# Observed vs modeled hydrograph ------------------------------------------

option <- 1
i <- 1
fname <- paste0("./data/results/exp", option, "/pred_", i, ".Rda")
load(fname)

data_plot <- pred_df %>%
  gather(cases, value, ob, pred) %>%
  mutate(cases = factor(cases, levels = c("ob", "pred"), labels = c("observation", "prediction")))

ggplot(data_plot, aes(datetime, value, color = cases, linetype = cases))+
  geom_line() +
  labs(y = "Outflow rate [CFS]") +
  theme_bw() +
  theme(legend.position = "top")


