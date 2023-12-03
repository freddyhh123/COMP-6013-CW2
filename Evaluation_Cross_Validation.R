Cross_Validate <- function(ml_model, dataset, num_of_iterations, num_of_folds){
  #Function that performs 10 fold cross validation over training dataset
  #
  #Args:
  # ml_model: Perceptron Model trained earlier
  # dataset: Training Dataset
  # num_of_iterations: number of epochs
  # num_of_folds: number of folds to perform cross validation
  #
  #Returns:
  # Accuracy accross each fold and average accuracy

  ################################################################
  #Stratified 10 - fold cross validation
  #Randomly shuffle the data
  rows_count <- nrow(dataset)
  dataset<-dataset[sample(rows_count),]
  
  
  accuracy_train <- vector() #Store training accuracy
  accuracy_test <- vector() #Store testing accuracy
  
  num_of_iterations <- num_of_iterations #Number of iterations to train 
  
  num_of_folds <- num_of_folds
  
  total_rows <- nrow(dataset)
  fold_size <- as.integer(total_rows/ num_of_folds) #Size of individual folds eg. 90/ 10 = 9 rows
  test_start_index <- 1
  for(fold_no in 1:10){
   # print(paste("Fold No#", fold_no))
    test_end_index <- fold_no * fold_size     #eg. for 2nd fold: 2 * 9 = 18 for dataset with 90 rows
    if(fold_no == num_of_folds){
      test_end_index <- total_rows            #for last fold take all remaining instances..eg. [..,9,9,12]
    }
    test_data <- dataset[test_start_index:test_end_index, ]   #eg. dataset[10-18] becomes 1 fold (test set)
    train_data <- dataset[-(test_start_index:test_end_index), ] #rows [1-9]+[19-90] becomes training set 
    
    test_start_index <- test_end_index + 1
    
    
    ##################################################
    #TRAINING 9 FOLDS
    
    dataset_train_features <- train_data[1:4]
    dataset_train_labels <- train_data[5]
    train_rows <- nrow(train_data)
    correct <-0
    total <- 0
    for(k in 1:num_of_iterations){
      for(i in 1:train_rows){
        actual_class <- as.character(dataset_train_labels[i,])
        feature_row<-matrix(dataset_train_features[i,], nrow = 1, ncol = 4)
        feature_row <- unlist(feature_row)
        predicted_class <- ml_model$train(actual_class, feature_row)
        if(actual_class == predicted_class){
          correct <- correct +1
        }
        total <- total + 1
        
      }
    }
    accuracy_train[fold_no] <- correct/ total
    
    
    
    ##################################################
    
    #############################################
    #test data - 10% - one fold of data
    dataset_test_features <- test_data[1:4]
    dataset_test_labels <- test_data[5]
    
    test_rows <- nrow(test_data)
    correct <-0
    total <- 0
    for(i in 1:test_rows){
      #print("-------------------------------------")
      #print(paste("Run #",k))
      actual_class <- as.character(dataset_test_labels[i,])
      feature_row<-matrix(dataset_test_features[i,], nrow = 1, ncol = 4)
      feature_row <- unlist(feature_row)
      predicted_class <- ml_model$predict(feature_row)
      #print(paste("Predicted = ", pred, "Actual = ", bv))
      if(actual_class == predicted_class){
        correct <- correct +1
      }
      total <- total + 1
      #print("-------------------------------------")
      
    }
    
    accuracy_test[fold_no] <- correct/ total
    print(paste("Accuracy in fold -",fold_no, " = ",round(accuracy_test[fold_no]*100, 2), '%'))
    #############################################
    
    
  }
  
  print(paste("Average Training Accuracy = ", round(mean(accuracy_train) * 100, 2), '%'))
  print(paste("Average Testing Accuracy = ", round(mean(accuracy_test) * 100, 2), '%'))
  ################################################################
  
}