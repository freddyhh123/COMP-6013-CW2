library(dplyr)
library(ggplot2)
library(gridExtra)

# Why a seperate file?
# this one is for final learning rate evaluation, the plotting is slightly different!

# Feed in data
model_data <- read.csv("final.csv")

model_data$accuracy_validation <- pmin(pmax(model_data$accuracy_validation, 0), 100)

# Plot for accuracy
accuracy_plot <- ggplot(model_data, aes(x = as.factor(learning_rate), y = accuracy_validation, group = as.factor(train_size), color = as.factor(train_size))) +
  geom_line(linewidth = 1) +
  geom_point(color = "black") +
  theme_minimal() +
  scale_x_discrete(name = "Learning Rate") +
  scale_y_continuous(limits = c(0, 100)) +
  ylab("Validation Accuracy (%)") +
  labs(title = "Validation Accuracy vs Learning Rate") +
  theme(legend.position = "none")

# Plot for F1 score
f1_plot <- ggplot(model_data, aes(x = as.factor(learning_rate), y = f1_validation, group = as.factor(train_size), color = as.factor(train_size))) +
  geom_line(linewidth = 1) +
  geom_point(color = "black") +
  theme_minimal() +
  scale_x_discrete(name = "Learning Rate") +
  scale_y_continuous(limits = c(0, 1)) +
  ylab("F1 Score") +
  labs(title = "F1 Score vs Learning Rate")+
  theme(legend.position = "none")

# Combine plots
combined_plot <- grid.arrange(accuracy_plot, f1_plot, ncol = 2)

# Print the combined plot
print(combined_plot)