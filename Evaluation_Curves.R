plot_learning_curve <- function(ml_model, train_dataset, validation_dataset, number_of_iterations = 10){
  #Function that plots learning curve for accuracy vs Training set size
  #
  #Args:
  # ml_model: Perceptron Model trained earlier
  # train_dataset: 2/3rd training set
  # validation_dataset: 1/3rd held out unseen data
  # number_of_iterations: number of epochs to train Perceptron
  #

  
  accuracy_training <- vector()
  accuracy_validation <- vector()
  num_of_iterations <- 10
  
  #1/10th sample size
  train_sample_size = nrow(train_dataset) / 10
  
  for(m in 1:10){
    upper_limit <- m*train_sample_size
    if(m*train_sample_size <=  nrow(train_dataset)){
      df_train_features <- train_dataset[1:upper_limit,1:4]
      df_train_labels <- train_dataset[1:upper_limit,5]
    }
    else{
      #If sample size goes beyond dataset size, take complete dataset as 100%
      df_train_features <- train_dataset[1:4]
      df_train_labels <- train_dataset[5]
    }
    train_rows <- nrow(df_train_features)

    correct <-0
    total <- 0
    for(k in 1:num_of_iterations){
      for(i in 1:train_rows){
        actual_class <- as.character(df_train_labels[i])
        feature_row <- matrix(df_train_features[i,], nrow = 1, ncol = 4)
        feature_row <- unlist(feature_row)
        predicted_class <- ml_model$train(actual_class, feature_row)
        if(actual_class == predicted_class){
          correct <- correct +1
        }
        total <- total + 1
      }
    }
    accuracy_training <- c(accuracy_training, correct / total)
    
  }
  library(ggplot2)
  loc_data <- data.frame(training_sample = seq(10,100,by=10), accuracy = accuracy_training)
  ggplot(loc_data) + geom_line(mapping = aes(x=training_sample, y = accuracy)) + ylim(0:1) +
    labs(x = 'training sample size (%)', y = 'Accuracy')
  return(mean(accuracy_training))
  
}

plot_learning_curve_epochs <- function(ml_model, train_dataset, validation_dataset){
  #Function that plots learning curve for accuracy vs number of epochs
  #
  #Args:
  # ml_model: Perceptron Model trained earlier
  # train_dataset: 2/3rd training set
  # validation_dataset: 1/3rd held out unseen data

  #
  accuracy_training <- vector()
  accuracy_validation <- vector()
  num_of_epochs <- c(10, 50,100, 1000)
  
  
  for(num_of_iterations in num_of_epochs){

    df_train_features <- train_dataset[1:4]
    df_train_labels <- train_dataset[5]

    train_rows <- nrow(df_train_features)
    
    correct <-0
    total <- 0
    for(k in 1:num_of_iterations){
      for(i in 1:train_rows){
        #print("-------------------------------------")
        #print(paste("Run #",k))
        actual_class <- as.character(df_train_labels[i,])
        feature_row<-matrix(df_train_features[i,], nrow = 1, ncol = 4)
        feature_row <- unlist(feature_row)
        predicted_class <- ml_model$train(actual_class, feature_row)
        #print(paste("Predicted = ", pred, "Actual = ", bv))
        if(actual_class == predicted_class){
          correct <- correct +1
        }
        total <- total + 1
        #print("-------------------------------------")
        
      }
    }
    accuracy_training <- c(accuracy_training, correct/ total)
    # print(paste("Training Accuracy = ", accuracy_training))
    
  }
  library(ggplot2);
  loc_data <- data.frame(number_of_epochs = c(0, num_of_epochs), accuracy = c(0, accuracy_training))
  ggplot(loc_data) + geom_line(mapping = aes(x=number_of_epochs, y = accuracy)) + ylim(0:1) +
    labs(x = 'number of epochs (iterations)', y = 'Accuracy')
}

plot_learning_curve_learning_Rates <- function(train_dataset, validation_dataset, num_of_epochs){
  #Function that plots learning curve for accuracy vs different learning rates
  #
  #Args:
  # ml_model: Perceptron Model trained earlier
  # train_dataset: 2/3rd training set
  # validation_dataset: 1/3rd held out unseen data
  # num_of_epochs: number of epochs to train Perceptron
  #
  accuracy_training <- vector()
  accuracy_validation <- vector()
  learning_rates <- c(0.1, 0.01, 0.001, 0.0001)
  
  for(learn_rate in learning_rates){
    ml_model <- Perceptron(learn_rate)
    
    df_train_features <- train_dataset[1:4]
    df_train_labels <- train_dataset[5]
    
    train_rows <- nrow(df_train_features)
    
    correct <-0
    total <- 0
    for(k in 1:num_of_epochs){
      for(i in 1:train_rows){
        actual_class <- as.character(df_train_labels[i,])
        feature_row<-matrix(df_train_features[i,], nrow = 1, ncol = 4)
        feature_row <- unlist(feature_row)
        predicted_class <- ml_model$train(actual_class, feature_row)
        if(actual_class == predicted_class){
          correct <- correct +1
        }
        total <- total + 1
        
      }
    }
    accuracy_training <- c(accuracy_training, correct/ total)
    
  }
  library(ggplot2);
  loc_data <- data.frame(num_learning_rates = c(0, learning_rates), accuracy = c(0, accuracy_training))
  ggplot(loc_data) + geom_line(mapping = aes(x=num_learning_rates, y = accuracy)) + ylim(0:1) +
    labs(x = 'Learning Rates', y = 'Accuracy')
}