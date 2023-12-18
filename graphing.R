library(dplyr)
library(ggplot2)
library(scales)
library(gridExtra)

# Feed in data
model_data <- read.csv("final.csv")

# Sort all data by learning rates
learning_rates <- model_data %>% group_by(learning_rate) %>% split(.,f = .$learning_rate, drop = TRUE)

# Our discrete scale labels
x_labels <- c(100, 1000, 2500, 5000)

# Initialising data collecion lists
validation_accuracies <- list()
f1_scores <- list()

for (learning_rate in 1:length(learning_rates)) {
  # Get all train sizes within the learning rates
  training_sizes <- learning_rates[[learning_rate]] %>%
    group_by(train_size) %>%
    split(., f = .$train_size, drop = TRUE)
  # Make sure they are ordered and formatted for the legend
  train_sizes_ordered <- sort(unique(model_data$train_size))
  legend_labels <- as.character(train_sizes_ordered)
  for (train_size in 1:length(training_sizes)) {
    row <- training_sizes[[train_size]]
    # Record some statistics for later plots
    validation_accuracies <- append(validation_accuracies, list(row$accuracy_validation))
    f1_scores <- append(f1_scores, list(row$f1_validation))
    name_row <- head(row,1)

    # Plot the 2 accuracies for this epoch size
    ggplot(row, aes(x = factor(epoch_size, levels = x_labels), group = 1)) +
        geom_line(aes(y = accuracy_validation, color = "deeppink1")) +
        geom_point(aes(y = accuracy_validation)) +
        geom_line(aes(y = (accuracy_testing), color = "deepskyblue1")) +
        geom_point(aes(y = (accuracy_testing))) +
        scale_y_continuous(limits = c(0, 100)) +
        scale_x_discrete(labels = x_labels) +
        xlab("Epoch Size") +
        ylab("Accuracy (%)") +
        labs(color = "Accuracy") +
        theme(legend.position = "right") +
        ggtitle(paste("Train and Validation accuracy for learning rate: ",name_row$learning_rate," and train size: ",name_row$train_size)) +
        scale_color_manual(labels = c("Validation","Train"),values = c("deepskyblue1", "deeppink1")) +
        theme(legend.title.align=0.5) 
    ggsave(filename = paste(name_row$learning_rate,"-",name_row$train_size,".png"))
  }

  # Reverse so the legend is a little more readable
  validation_accuracies <- rev(validation_accuracies)
  f1_scores <- rev(f1_scores)

  # Format it all for ggplot
  accuracy_data <- data.frame(
    x = factor(rep(x_labels, length(validation_accuracies)), levels = x_labels),
    y = unlist(validation_accuracies),
    group = rep(1:length(validation_accuracies), each=length(x_labels))
  )
  f1_data <- data.frame(
    x = factor(rep(x_labels, length(f1_scores)), levels = x_labels),
    y = unlist(f1_scores),
    group = rep(1:length(f1_scores), each=length(x_labels))
  )

  # Plot the accuracies for this learning rate
  accuracy_plot <- ggplot(accuracy_data, aes(x=x, y=y, group=group, color=as.factor(group))) +
    geom_line(linewidth = 1) +
    geom_point() +
    theme_minimal() +
    scale_x_discrete(name = "Epochs") +
    scale_y_continuous(limits = c(0, 100)) +
    ylab("Accuracy (%)") +
    scale_colour_manual(values = c("red", "green", "blue", "orange", "purple"), 
        labels = legend_labels, 
        name = "Train size") +
    labs(title = "Line Plot of Validation Accuracies", color = "Group")

  # Plot the F1s for this learning rate
  f1_plot <- ggplot(f1_data, aes(x=x, y=y, group=group, color=as.factor(group))) +
    geom_line(linewidth = 1) +
    geom_point() +
    theme_minimal() +
    scale_x_discrete(name = "Epochs") +
    scale_y_continuous(limits = c(0, 1)) +
    ylab("F1 score") +
    scale_colour_manual(values = c("red", "green", "blue", "orange", "purple"), 
        labels = legend_labels, 
        name = "Train size") +
    labs(title = "Line Plot of F1 scores", color = "Group")

  # Combine for easy viewing
  combined_plot <- grid.arrange(accuracy_plot, f1_plot, ncol=2)

}
ggsave("combined_plot.png", combined_plot, width = 16, height = 8)
