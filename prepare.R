# Libraries ---------------------------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, lubridate, RcppRoll, zeallot, xgboost)


# Generate random rainfalls -----------------------------------------------

set.seed(10000)

n <- 40
base_shape <- c(1,2,6,2,1)

rain_locations <- runif(n, min = 0, max = 4500) %>% round
rain_durations <- runif(n, min = 5, max = 60) %>% round
rain_peaks <- rnorm(n, 0.5, 0.8) %>% abs()

out <- rep(0, 5000)
for (i in 1:n){
  
  rain_series <- approx(x = 1:5, base_shape, n = rain_durations[i])$y/6*rain_peaks[i]
  
  out[rain_locations[i]:(rain_locations[i] + rain_durations[i] - 1)] <-   out[rain_locations[i]:(rain_locations[i] + rain_durations[i] - 1)]  +
    rain_series
}

out <- tibble(
  datetime = seq(ymd_hm("2007-01-01 00:00"), by = 600, length.out = length(out)),
  rain = out
)

# visualize rainfall time series
ggplot(out, aes(datetime, rain)) +
  geom_line() +
  labs(y = "Rainfall intensity [Inch/hour]")

# write 
write.table(
  out %>%
    mutate(lines = paste("rain", year(datetime), month(datetime), day(datetime), hour(datetime), minute(datetime), rain)) %>%
    pull(lines),
  file = "./data/rain.data",
  row.names = F, 
  quote = F, 
  col.names = F
  )

# Runoff time series ------------------------------------------------------

runoff <- read.table("./data/outflow.txt", skip = 9) %>%
  as_tibble() %>%
  transmute(j = "SWMM",
            datetime = ymd_hms(paste(V2, V3, V4, V5, V6, V7)),
            runoff = V8) %>%
  filter(datetime %in% out$datetime)

ggplot(runoff, aes(datetime, runoff)) +
  geom_line() +
  labs(y = "Flow rate [CFS]")


# save rainfall-runoff data
rainfall_runoff <- out %>%
  left_join(runoff, by = "datetime") %>%
  select(datetime, X = rain, y = runoff) %>%
  .[complete.cases(.),]

write.csv(rainfall_runoff, 
          file = "./data/rainfall_runoff.data",
          row.names = F)
