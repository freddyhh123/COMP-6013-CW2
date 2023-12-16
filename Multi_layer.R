library(dplyr)
library(ggplot2)

one_hot_encode <- function(vector) {
  # Find unique categories
  categories <- unique(vector)

  # Initialize matrix for one-hot encoded vectors
  one_hot_matrix <- matrix(0, nrow = length(vector), ncol = length(categories))

  # Assign column names for readability
  colnames(one_hot_matrix) <- categories

  # Fill the matrix with 1's where appropriate
  for (i in 1:length(vector)) {
    category_index <- which(categories == vector[i])
    one_hot_matrix[i, category_index] <- 1
  }

  return(one_hot_matrix)
}

decode_one_hot <- function(vector) {
  return(max.col(vector))
}

# Define the softmax function
softmax <- function(z) {
  exp_z <- exp(z) #np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
  softmax_output <- exp_z / sum(exp_z)
  return(softmax_output)
}

# Define the derivative of the softmax function
softmax_derivative <- function(softmax_output, target) {
  softmax_output - target
}

leaky_relu <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, x, alpha * x))
}
leaky_relu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

standardize_data <- function(data) {
  # Ensure data is a dataframe or matrix
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop("Input must be a dataframe or matrix.")
  }

  standardized_data <- as.data.frame(lapply(data, function(x) {
    if (is.numeric(x)) {
      (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
    } else {
      x  # Skip non-numeric columns
    }
  }))

  return(standardized_data)
}

cross_entropy_loss <- function(predictions, labels) {
  return(-logb(predictions[which.max(labels)]))
}

# Define the function to initialize weights and biases
initialize_weights <- function(input_size, hidden_size, output_size) {
  list(
    input_weights = matrix(rnorm(input_size * hidden_size, mean = 0, sd = sqrt(2 / input_size)), input_size, hidden_size),
    hidden_biases = matrix(0, 1, hidden_size),
    hidden_weights = matrix(rnorm(hidden_size * output_size, mean = 0, sd = sqrt(1 / hidden_size)), hidden_size, output_size),
    output_biases = matrix(0, 1, output_size)
  )
}

# Define the forward pass function
forward_pass <- function(inputs, weights) {
  # Compute hidden layer output
  hidden_linear_combination <- inputs %*% t(weights$input_weights) + weights$hidden_biases
  hidden_layer_output <- leaky_relu(hidden_linear_combination)

  # Compute output layer input and output
  output_layer_input <- hidden_layer_output %*% weights$hidden_weights + weights$output_biases
  output_layer_output <- softmax(output_layer_input)

  list(
    hidden_layer_output = hidden_layer_output,
    output_layer_output = output_layer_output
  )
}

backpropagation <- function(inputs, actual_output, weights, learning_rate, epoch) {
  forward_pass_results <- forward_pass(inputs, weights)

  loss = cross_entropy_loss(forward_pass_results$output_layer_output, actual_output)
  
  #plot(loss_list$epoch, loss_list$loss, type = 'l')

  # Unpack the forward pass results
  hidden_layer_output <- forward_pass_results$hidden_layer_output
  output_layer_output <- forward_pass_results$output_layer_output

  # Error at the output layer
  output_error <- output_layer_output - actual_output

  # Gradient calculation for output layer
  output_weights_gradient <- t(hidden_layer_output) %*% as.numeric(output_error)
  output_biases_gradient <- colSums(output_error)

  # Backpropagate the error to the hidden layer
  hidden_error <- (weights$hidden_weights %*% matrix(as.numeric(output_error), ncol = 1)) * as.numeric(leaky_relu_derivative(hidden_layer_output))
  print(as.numeric(leaky_relu_derivative(hidden_layer_output)))
  # Gradient calculation for hidden layer
  # Ensure inputs is a row vector for multiplication
  input_weights_gradient <- matrix(inputs, nrow = 1) %*% hidden_error
  input_biases_gradient <- colSums(hidden_error)

  # Update weights and biases
  weights$input_weights <- weights$input_weights - learning_rate * as.numeric(input_weights_gradient)
  weights$hidden_biases <- weights$hidden_biases - learning_rate * input_biases_gradient
  weights$hidden_weights <- weights$hidden_weights - learning_rate * output_weights_gradient
  weights$output_biases <- weights$output_biases - learning_rate * output_biases_gradient

  return(list(weights = weights, loss = loss))
}

# Define the training function
train_mlp <- function(inputs, targets, hidden_size, learning_rate, num_epochs) {
  input_size <- ncol(inputs)

  weights <- initialize_weights(input_size, hidden_size, 3)

  loss_frame <- data.frame(
        loss = numeric(),
        epoch = numeric()
  )
  for (epoch in 1:100) {
    for (i in 1:nrow(inputs)) {
      input_data <- inputs[i, , drop = FALSE]
      target_data <- targets[i,]
      results <- backpropagation(input_data, target_data, weights, 0.01, epoch)
      weights <- results$weights
      loss <- results$loss
      if(is.na(loss)){
        browser()
      }
      loss_frame[nrow(loss_frame) + 1, ] = c(loss,epoch)
      print(paste("loss: ",loss," Epoch: ",epoch))
    }
  }
  return(list(weights = weights, loss = loss_frame))
}

train_sizes <- c(40, 50, 60, 70, 80)
learning_rates <- c(0.001, 0.01, 0.25, 0.5)
epoch_sizes <- c(100, 500, 1000, 2500, 5000)

set.seed(042)
penguins <- palmerpenguins::penguins_raw

penguins <- penguins[, c(3, 10, 11, 12, 13, 14)]

penguins <- na.omit(penguins)

penguins_standardised <- standardize_data(penguins[2:5])

penguinsMove <- penguins[, 1]
penguinsMove <- cbind(penguins_standardised, penguinsMove)

penguins <- penguinsMove

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



for (train_size in train_sizes) {
  for (learning_rate in learning_rates) {
    rows_count <- nrow(penguins)

    for(k in 1:5){
      penguins <- penguins[sample(rows_count),]
    }

    one_hot <- one_hot_encode(penguins[,5])

    penguins <- cbind(penguins,one_hot)

    source("Perceptron.r")
    source("Evaluation_Cross_Validation.r")
    source("Evaluation_Validation.r")
    source("Evaluation_Curves.r")
    validation_instances <- sample(nrow(penguins)*(train_size/100))
    penguins_validation <- penguins[validation_instances, ]
    penguins_train <- penguins[-validation_instances, ]
    cat("\nPenguin_validation size:\n")
    print(nrow(penguins_validation))
    cat("\nPenguin_train size:\n")
    print(nrow(penguins_train))

    predictions <- data.frame(
      actual = numeric(),
      predict = numeric()
    )

    for(epoch_size in epoch_sizes){
      row <- c(train_size, learning_rate, epoch_size)
      results <- train_mlp(as.matrix(penguins_train[, 1:4]),penguins_train[,6:8], 4, 0.001, epoch_size)
      mlp_weights <- results$weights
      loss <- results$loss

      for(row in validation_instances){
        actual <- as.matrix(penguins_validation[row,][5:8])
        actual <- as.numeric(actual[2:4])
        prediction <- forward_pass(as.matrix(penguins_validation[row,][1:4]), mlp_weights)$output_layer_output
        max_index <- which.max(prediction)
        one_hot_encoded <- rep(0, length(prediction))
        one_hot_encoded[max_index] <- 1

        predictions <- rbind(predictions, data.frame(actual = which.max(actual), predict= which.max(prediction)))
      }
      saveRDS(mlp_weights, paste(train_size,learning_rate,epoch_size,"mlp_weights.rds", sep = "-"))
      average_loss_per_epoch <- loss %>%
        group_by(epoch) %>%
        summarise(average_loss = mean(loss))

      epoch_plot -> ggplot(average_loss_per_epoch, aes(x = epoch, y = average_loss)) +
        geom_line() +   # for a line plot
        labs(title = "Average Loss per Epoch", x = "Epoch", y = "Average Loss") +
        theme_minimal()
    }

    # Evaluate and collect results
    model_accuracy[count, ] <- row
    write.csv(model_accuracy, "./main.csv", row.names = FALSE)
  }
}

