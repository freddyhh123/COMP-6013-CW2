library(dplyr)
library(ggplot2)
library(scales)

model_data <- read.csv("main.csv")

learning_rates <- model_data %>% group_by(learning_rate) %>% split(.,f = .$learning_rate, drop = TRUE)

x_labels <- c(100, 500, 1000, 2500, 5000)

for (learning_rate in 1:length(learning_rates)) {
  training_sizes <- learning_rates[[learning_rate]] %>% group_by(train_size) %>% split(.,f = .$train_size, drop = TRUE)
  for (train_size in 1:length(training_sizes)) {
    row <- training_sizes[[train_size]]
    name_row <- head(row,1)
    ggplot(row, aes(x = factor(epoch_size, levels = x_labels), group = 1)) +
        geom_line(aes(y = accuracy_validation, color = "deepskyblue1")) +
        geom_point(aes(y = accuracy_validation)) +
        geom_line(aes(y = (accuracy_testing*100), color = "deeppink1")) +
        geom_point(aes(y = (accuracy_testing*100))) +
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
}
