library(dplyr)
library(ggplot2)

source("Perceptron.r")
source("Evaluation_Cross_Validation.r")
source("Evaluation_Validation.r")
source("Evaluation_Curves.r")

one_hot_encode <- function(Categories) {
  # Get our unique labels
  categories <- unique(Categories)

  # Matrix for encoding
  one_hot_matrix <- matrix(0, nrow = length(Categories), ncol = length(categories))

  # Assign column names
  colnames(one_hot_matrix) <- categories

  # Fill the matrix
  for (i in 1:length(Categories)) {
    category_index <- which(categories == Categories[i])
    one_hot_matrix[i, category_index] <- 1
  }

  return(one_hot_matrix)
}

decode_one_hot <- function(one_hot) {
  return(max.col(one_hot))
}

# Define the softmax function
softmax <- function(x) {
  exp_x <- exp(x)
  softmax_output <- exp_x / sum(exp_x)
  return(softmax_output)
}

# Using Leaky relu to allow some leeway in the variables
# there were issues with dead neurons and this helped!
# same as relu but uses a very small alpha instead of 0
# for negative values
leaky_relu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}
leaky_relu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

# Here we are using Z score normalisation as min-max and mean
# normalisation were causing issues with very small numbers
standardize_data <- function(data) {
  # Make sure data is a dataframe or matrix
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop("Not dataframe or matrix.")
  }

  standardized_data <- as.data.frame(lapply(data, function(x) {
    # Checking for numerical columns
    if (is.numeric(x)) {
      (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
    } else {
      x 
    }
  }))

  return(standardized_data)
}

# Cross entropy loss function, this allows us to 
# evaluate model performance
cross_entropy_loss <- function(predictions, labels) {
  return(-logb(predictions[which.max(labels)]))
}

# Define the function to initialize weights and biases
initialize_weights <- function(input_size, hidden_size1, hidden_size2, hidden_size3, output_size) {
  list(
    # For the relu layers we are using He initialisation, this has a provable increase on model performance
    input_weights = matrix(rnorm(input_size * hidden_size1, mean = 0, sd = sqrt(2 / input_size)), input_size, hidden_size1),
    hidden_biases1 = matrix(0, 1, hidden_size1),
    hidden_weights1 = matrix(rnorm(hidden_size1 * hidden_size2, mean = 0, sd = sqrt(2 / hidden_size1)), hidden_size1, hidden_size2),
    hidden_biases2 = matrix(0, 1, hidden_size2),
    hidden_weights2 = matrix(rnorm(hidden_size2 * hidden_size3, mean = 0, sd = sqrt(2 / hidden_size2)), hidden_size2, hidden_size3),
    hidden_biases3 = matrix(0, 1, hidden_size3),
    hidden_weights3 = matrix(rnorm(hidden_size3 * output_size, mean = 0, sd = sqrt(2 / hidden_size3)), hidden_size3, output_size),
    output_biases = matrix(0, 1, output_size)
  )
}
# Define the forward pass function
forward_pass <- function(inputs, weights) {
  # First hidden layer
  hidden1_linear_combination <- inputs %*% t(weights$input_weights) + weights$hidden_biases1
  hidden1_output <- leaky_relu(hidden1_linear_combination)

  # Second hidden layer
  hidden2_linear_combination <- hidden1_output %*% weights$hidden_weights1 + weights$hidden_biases2
  hidden2_output <- leaky_relu(hidden2_linear_combination)

  # Third hidden layer
  hidden3_linear_combination <- hidden2_output %*% weights$hidden_weights2 + weights$hidden_biases3
  hidden3_output <- leaky_relu(hidden3_linear_combination)

  # Output layer
  output_layer_input <- hidden3_output %*% weights$hidden_weights3 + weights$output_biases
  output_layer_output <- softmax(output_layer_input)

  list(
    hidden1_output = hidden1_output,
    hidden2_output = hidden2_output,
    hidden3_output = hidden3_output,
    output_layer_output = output_layer_output
  )
}

backpropagation <- function(inputs, actual_output, weights, learning_rate, epoch) {
  forward_pass_results <- forward_pass(inputs, weights)

  # Calculate our loss for model evaluation
  loss <- cross_entropy_loss(forward_pass_results$output_layer_output, actual_output)
  # Check if we got the correct answer (again for model performance)
  accuracy <- list(prediction = decode_one_hot(forward_pass_results$output_layer_output), label = decode_one_hot(actual_output))

  # Move out of the list for simplified referencing
  hidden1_output <- forward_pass_results$hidden1_output
  hidden2_output <- forward_pass_results$hidden2_output
  hidden3_output <- forward_pass_results$hidden3_output
  output_layer_output <- forward_pass_results$output_layer_output

  # Error at the output layer
  output_error <- output_layer_output - actual_output

  # Gradient calculation for output layer
  output_weights_gradient <- t(hidden3_output) %*% as.numeric(output_error)
  output_biases_gradient <- colSums(output_error)

  # Backpropagate the error to the third hidden layer
  hidden3_error <- (weights$hidden_weights3 %*% matrix(as.numeric(output_error), ncol = 1)) * as.numeric(leaky_relu_derivative(hidden3_output))

  # Gradient calculation for third hidden layer
  hidden3_weights_gradient <- t(hidden2_output) %*% as.numeric(hidden3_error)
  hidden3_biases_gradient <- colSums(hidden3_error)

  # Backpropagate the error to the second hidden layer
  hidden2_error <- (weights$hidden_weights2 %*% matrix(as.numeric(hidden3_error), ncol = 1)) * as.numeric(leaky_relu_derivative(hidden2_output))

  # Gradient calculation for second hidden layer
  hidden2_weights_gradient <- t(hidden1_output) %*% as.numeric(hidden2_error)
  hidden2_biases_gradient <- colSums(hidden2_error)

  # Backpropagate the error to the first hidden layer
  hidden1_error <- (weights$hidden_weights1 %*% hidden2_error) * as.numeric(leaky_relu_derivative(hidden1_output))

  # Gradient calculation for first hidden layer
  input_weights_gradient <- matrix(inputs, nrow = 1) %*% as.numeric(hidden1_error)
  input_biases_gradient <- colSums(hidden1_error)

  # Update weights and biases
  weights$input_weights <- weights$input_weights - learning_rate * as.numeric(input_weights_gradient)

  weights$hidden_biases1 <- weights$hidden_biases1 - learning_rate * input_biases_gradient
  weights$hidden_weights1 <- weights$hidden_weights1 - learning_rate * hidden2_weights_gradient

  weights$hidden_biases2 <- weights$hidden_biases2 - learning_rate * hidden2_biases_gradient
  weights$hidden_weights2 <- weights$hidden_weights2 - learning_rate * hidden3_weights_gradient

  weights$hidden_biases3 <- weights$hidden_biases3 - learning_rate * hidden3_biases_gradient
  weights$hidden_weights3 <- weights$hidden_weights3 - learning_rate * output_weights_gradient

  weights$output_biases <- weights$output_biases - learning_rate * output_biases_gradient

  # Return all our info for unpacking and analysis later
  return(list(weights = weights, loss = loss, accuracy = accuracy))
}

# Define the training function
train_mlp <- function(inputs, targets, hidden_size, learning_rate, num_epochs) {
  input_size <- ncol(inputs)
  output_size <- 3
  # Initialise our weights
  weights <- initialize_weights(input_size, hidden_size, hidden_size, hidden_size, output_size)

  # Initialising lists for data collection
  # Average loss for this epoch size
  loss_frame <- data.frame(
    loss = numeric(),
    epoch = numeric()
  )
  # Average accuracy for this epoch size
  accuracy_averages <- list()
  for (epoch in 1:epoch_size) {
    if(epoch %% 50 == 0){
      print(paste("At Epoch:",epoch))
    }
    training_accuracy <- data.frame(
      predicted = numeric(),
      actual = numeric()
    )
    for (i in 1:nrow(inputs)) {
      # Seperate out our input and target data
      input_data <- inputs[i, , drop = FALSE]
      target_data <- targets[i, ]
      # Forward propigation is within the backpropigation function
      results <- backpropagation(input_data, target_data, weights, learning_rate, epoch)
      # Unpack our results
      weights <- results$weights
      loss <- results$loss
      accuracy <- results$accuracy
      loss_frame[nrow(loss_frame) + 1, ] <- c(loss, epoch)
      training_accuracy[nrow(training_accuracy) + 1, ] <- accuracy
    }
    # Get the accuracy averages for this epoch
    accuracy_averages[epoch] <- sum(training_accuracy$predicted == training_accuracy$actual, na.rm = TRUE) / nrow(training_accuracy) * 100
  }
  return(list(weights = weights, loss = loss_frame, accuracy = mean(unlist(accuracy_averages))))
}


# Set our lists to iterate through
train_sizes <- c(40, 50, 60, 70, 80)
learning_rates <- c(0.001, 0.01, 0.25, 0.5)
epoch_sizes <- c(100, 1000, 2500, 5000)
hidden_size <- 4

# Set our seed
set.seed(042)

# Seperate our data and standardize!
penguins <- palmerpenguins::penguins_raw

penguins <- penguins[, c(3, 10, 11, 12, 13, 14)]

penguins <- na.omit(penguins)

penguins_standardised <- standardize_data(penguins[2:5])

penguinsMove <- penguins[, 1]
penguinsMove <- cbind(penguins_standardised, penguinsMove)

penguins <- penguinsMove

# Setup the dataframe for logging
model_accuracy <- data.frame(
  train_size = numeric(),
  learning_rate = numeric(),
  epoch_size = numeric(),
  accuracy_testing = numeric(),
  accuracy_validation = numeric(),
  f1_validation = numeric(),
  precision = numeric(),
  recall = numeric()
)


print("Started!")
for (train_size in train_sizes) {
  for (learning_rate in learning_rates) {
    rows_count <- nrow(penguins)

    for (k in 1:5) {
      penguins <- penguins[sample(rows_count), ]
    }

    # One hot encode our labels
    one_hot <- one_hot_encode(penguins[, 5])

    penguins <- cbind(penguins, one_hot)

    # Seperate out our validation instances (using the train_size)
    validation_instances <- sample(nrow(penguins) * (train_size / 100))
    penguins_validation <- penguins[validation_instances, ]
    penguins_train <- penguins[-validation_instances, ]
    cat("\nPenguin_validation size:\n")
    print(nrow(penguins_validation))
    cat("\nPenguin_train size:\n")
    print(nrow(penguins_train))

    # Setup data frames for data collection
    predictions <- data.frame(
      actual = numeric(),
      predict = numeric()
    )
    epoch_data <- data.frame(
      train_size = numeric(),
      learning_rate = numeric(),
      epoch_size = numeric(),
      average_training_accuracy = numeric(),
      average_validation_accuracy = numeric(),
      f1_validation = numeric(),
      precision = numeric(),
      recall = numeric(),
      average_loss = numeric()
    )
    for (epoch_size in epoch_sizes) {
      # Confusion Matrix
      distinct_classes <- c(1, 2, 3)
      # Row ids are actual classes
      row_id <- 0
      # Columns are predicted classes
      col_id <- 0
      # Confusion Matrix
      confusion_matrix <- matrix(numeric(length(distinct_classes)^2),
        nrow = length(distinct_classes),
        ncol = length(distinct_classes)
      )
      row.names(confusion_matrix) <- distinct_classes
      colnames(confusion_matrix) <- distinct_classes

      # Train our model!
      results <- train_mlp(as.matrix(penguins_train[, 1:4]), penguins_train[, 6:8], hidden_size, learning_rate, epoch_size)
      mlp_weights <- results$weights
      loss <- results$loss

      # Go through our validation set
      for (row in validation_instances) {
        #Get the "real" value and our model's guess
        actual <- as.matrix(penguins_validation[row, ][5:8])
        actual <- as.numeric(actual[2:4])

        prediction <- forward_pass(as.matrix(penguins_validation[row, ][1:4]), mlp_weights)$output_layer_output
        guess <- which.max(prediction)

        # Fill in the confusion matrix
        row_id <- which(distinct_classes == which.max(actual))
        col_id <- which(distinct_classes == guess)
        confusion_matrix[row_id, col_id] <- confusion_matrix[row_id, col_id] + 1

        predictions <- rbind(predictions, data.frame(actual = which.max(actual), predict = guess))
      }
      # Save the weights (just incase!)
      saveRDS(mlp_weights, paste(train_size, learning_rate, epoch_size, "mlp_weights.rds", sep = "-"))
      # Calculate average loss for each epoch (and plot)
      average_loss_per_epoch <- loss %>%
        group_by(epoch) %>%
        summarise(average_loss = mean(loss))

      ggplot(average_loss_per_epoch, aes(x = epoch, y = average_loss)) +
        geom_line(color = "darkblue", size = 1, linetype = "solid") +
        labs(x = "Epoch", y = "Average Loss") +
        theme_minimal() +
        ggtitle(paste("Average loss per Epoch:", learning_rate, "and train size:", train_size))
      ggsave(filename = paste(epoch_size, "-", learning_rate, "-", train_size, ".png"))

      correct_predictions <- sum(predictions$actual == predictions$predict, na.rm = TRUE) / nrow(predictions) * 100

      # Calculate our F1 score
      TP <- confusion_matrix[1, 1] + confusion_matrix[2, 2] + confusion_matrix[3, 3]
      FP <- confusion_matrix[1, 2] + confusion_matrix[1, 3] + confusion_matrix[2, 1]
      FN <- confusion_matrix[2, 3] + confusion_matrix[3, 2] + confusion_matrix[3, 1]
      precision <- TP / (TP + FP)
      recall <- TP / (TP + FN)
      f1_score <- (precision * recall) / 2

      # This is our main information row, its what will be saved to a CSV
      epoch_data[nrow(epoch_data) + 1, ] <- c(train_size, learning_rate, epoch_size, results$accuracy, correct_predictions, f1_score, precision, recall, mean(average_loss_per_epoch$average_loss))
      print(paste("Epoch Size:",epoch_size, "Complete!"))
      print(paste("Training Accuracy:",results$accuracy,"Validation accuracy:", correct_predictions))
      print(paste("F1 Score:", f1_score))
      print(paste("Loss:", mean(average_loss_per_epoch$average_loss)))
      print(paste("Epochs completed:",nrow(epoch_data),"/",length(epoch_sizes)))
      cat("\n")
    }
    print(paste("Learning Rate:",learning_rate, "Complete!"))
    cat("\n")
    write.csv(epoch_data, "./main_MLP.csv", row.names = FALSE)
    print(paste("CSV Written!"))
    cat("\n")
  }
    print(paste("Train Size:",train_size, "Complete!"))
    cat("\n")
}
