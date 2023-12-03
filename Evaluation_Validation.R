Validate <- function(ml_model, train_dataset, validation_dataset, number_of_iterations = 10){
  #Function that computes accuracy, confusion matrix and precison over unseen validation dataset
  #
  #Args:
  # ml_model: Perceptron Model trained earlier
  # train_dataset: 2/3rd training set
  # validation_dataset: 1/3rd held out unseen data
  # number_of_iterations: number of epochs to train Perceptron
  #
  accuracy_training <- vector()
  accuracy_validation <- vector()
  num_of_iterations <- number_of_iterations
  
  df_train_features <- train_dataset[1:4]
  df_train_labels <- train_dataset[5]
  train_rows <- nrow(train_dataset)
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
  accuracy_training <- correct/ total
  #print(paste("Training Accuracy = ", accuracy_training))
  
  ####################################################################
  #Testing over Validation set
  
  df_valid_features <- validation_dataset[1:4]
  df_valid_labels <- validation_dataset[5]
  
  #Confusion Matrix
  distinct_classes <- as.vector(unlist(unique(df_valid_labels)))
  #Row ids are actual classes
  row_id <- 0
  #Columns are predicted classes
  col_id <- 0
  #Confusion Matrix
  confusion_matrix <- matrix(numeric(length(distinct_classes)^2), 
                             nrow = length(distinct_classes), 
                             ncol = length(distinct_classes))
  
  row.names(confusion_matrix) <- distinct_classes
  colnames(confusion_matrix) <- distinct_classes
  
  valid_rows <- nrow(validation_dataset)
  correct <-0
  total <- 0
  for(i in 1:valid_rows){
    #print("-------------------------------------")
    #print(paste("Run #",k))
    actual_class <- as.character(df_valid_labels[i,])
    feature_row<-matrix(df_valid_features[i,], nrow = 1, ncol = 4)
    feature_row <- unlist(feature_row)
    predicted_class <- ml_model$predict(feature_row)
    #print(paste("Predicted = ", pred, "Actual = ", bv))
    if(actual_class == predicted_class){
      correct <- correct +1
    }
    total <- total + 1
    #print("-------------------------------------")
    #Confusion Matrix
    row_id <- which(distinct_classes == actual_class)
    col_id <- which(distinct_classes == predicted_class)
    confusion_matrix[row_id, col_id] <- confusion_matrix[row_id, col_id] + 1
  }

  TP <- confusion_matrix[1, 1] + confusion_matrix[2, 2] + confusion_matrix[3, 3]
  FP <- confusion_matrix[1, 2] + confusion_matrix[1, 3] + confusion_matrix[2, 1]
  FN <- confusion_matrix[2, 3] + confusion_matrix[3, 2] + confusion_matrix[3, 1]

  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- (precision * recall) / 2

  accuracy_validation <- correct/ total
  cat(paste("\nAccuracy over validation set (unseen data) = ", round(accuracy_validation * 100, digits = 2), "%"))
  cat("\n--------------------------------------------------")
  cat("\nConfusion Matrix: Actual vs Predicted classes:\n")
  print(confusion_matrix)
  cat("\n--------------------------------------------------")
  cat("\nPrecision (%) in classification of different classes:\n")
  sapply(distinct_classes, function(category){
    precison <- (confusion_matrix[category,category]*100)/ sum(confusion_matrix[category,]);
  })
  cat("\nF1 score:\n")
  print(f1_score)

  return(c(round(accuracy_validation * 100, digits = 2), f1_score,precision,recall))
}