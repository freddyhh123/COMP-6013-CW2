library(dplyr)

data <- read.csv("main copy 2.csv")

data <- data %>% 
  mutate(F1_Score = 2 * (precision * recall) / (precision + recall))

print(data$F1_Score)