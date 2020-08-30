# Libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, lubridate, RcppRoll, zeallot, xgboost)

# Functions ---------------------------------------------------------------


# Data --------------------------------------------------------------------

read_csv("./data/rainfall_runoff.data")
