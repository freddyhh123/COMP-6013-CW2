#Main File to import dataset, train, test, validate and plot learning curves
set.seed(042)
penguins <- palmerpenguins::penguins_raw

penguins <- penguins[, c(3, 10, 11, 12, 13, 14)]

penguins <- na.omit(penguins)

#penguins[, c(2, 3, 4, 5)] <- scale(penguins[, c(2, 3, 4, 5)], scale=TRUE)

penguinsMove <- penguins[, 1]
penguinsMove <- cbind(penguins[2:5], penguinsMove)

penguins <- penguinsMove

train_sizes <- c(30, 40, 50, 60, 70, 80)
learning_rates <- c(0.001, 0.01, 0.1, 0.25, 0.5)
epoch_sizes <- c(100, 1000, 10000)


models <- vector(length = length(train_sizes))
columns = c("train_size", "learning_rate", "epoch_size", "accuracy", "f1","precision","recall")
model_accuracy <- data.frame(matrix(nrow = (length((train_sizes))*length(learning_rates)*length(epoch_sizes))), ncol = 7)


for (train_size in 1:length(train_sizes)) {
  for (learning_rate in 1:length(learning_rates)) {
    for (epoch_size in 1:length(epoch_sizes)) {
      #Randomly shuffle the dataset rows (repeatedly shuffled for 5 times)
      rows_count <- nrow(penguins)

      for(k in 1:5){
        penguins <- penguins[sample(rows_count),]
      }

      source("Perceptron.r")
      source("Evaluation_Cross_Validation.r")
      source("Evaluation_Validation.r")
      source("Evaluation_Curves.r")


      #Hold out 1/3 rd validation dataset
      validation_instances <- sample(nrow(penguins)*(train_sizes[train_size]/100))
      penguins_validation<-penguins[validation_instances,]
      penguins_train <- penguins[-validation_instances,]
      cat("\nPenguin_validation size:\n")
      print(nrow(penguins_validation))
      cat("\nPenguin_train size:\n")
      print(nrow(penguins_train))


      #Build Perceptron Model
      p_model <- Perceptron(learning_rates[learning_rate])


      #Set number of epochs (iterations)
      num_of_epochs <- epoch_sizes[epoch_size]  #Ideally, run with 1000 number of epochs but 1000 takes considerable amount (>10 min) to train

      #plot Learning Curve - Accuracy vs Training Sample size
      plot_learning_curve(p_model, penguins_train, penguins_validation, number_of_iterations = num_of_epochs)

      #plot Learning Curve - Accuracy vs Number of Epochs (Iterations)
      plot_learning_curve_epochs(p_model, penguins_train, penguins_validation)

      #plot Learning Curve - Accuracy vs Learning Rate values
      plot_learning_curve_learning_Rates(penguins_train, penguins_validation, num_of_epochs = num_of_epochs)

      #Train - Test - Cross Validate accross 10 folds
      Cross_Validate(p_model, penguins_train, num_of_iterations = num_of_epochs, num_of_folds = 10)
      #Cross_Validate(ml_model, dataset, num_of_iterations, num_of_folds)

      #Validate results with held out validation dataset
      row <- c(train_sizes[train_size], learning_rates[learning_rate], epoch_sizes[epoch_size], Validate(p_model, penguins_train, penguins_validation, number_of_iterations = 10))
      print(row)
      model_accuracy[nrow(model_accuracy) + 1, ] <- row
    }
  }
}
1<-1
