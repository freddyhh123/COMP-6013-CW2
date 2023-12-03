
Perceptron <- function(learning_rate){
  #Closure that contains functions for activation, backpropagation, 
  #training and testing a perceptron
  
  
  #List of weights of different classes,
  #This list will contain array of weights corresponding to each class
  label_weights <- list()
  
  #Set Learning rate
  learning_rate <- learning_rate;
  
  decide_class <- function(feature_set){
    #Function that decides output class value based upon weighted sum of inputs
    #
    #Args:
    # feature_set: set of input features for a single instance of data (training/ testing set)
    #
    #Returns:
    # Outputs class label with highest weighted sum

    #Compute weighted sum of inputs and synaptic weights (dot product of weights and inputs) 
    # for each class, and find the class with maximum value
    
    names(which.max(lapply(label_weights, matrix_dot_product, feature_set)))
  }

  matrix_dot_product <- function(weight_matrix, feature_matrix){
    #Function that computes dot product of synaptic weights and input features
    #
    #Args:
    # weight_matrix: set of perceptron synaptic weights corresponding to each class
    # feature_matrix: set of features (dimensions)
    #
    #Returns:
    # Dot product of weights and dimensions matrices (weighted sum)
    weight_matrix = unlist(matrix(weight_matrix, 1, 4))
    #Dot product of Transpose([n x 1]) (dot) [n x 1] matrices yields [1 x 1] scalar (weighted sum)

    #t(feature_matrix) %*% weight_matrix
    feature_matrix %*% t(weight_matrix)
  }

  
  update_synaptic_weights <- function(weight_matrix, error, feature_matrix){
    #Function that updates synaptic weights based on backpropagation error and learning rate
    #
    #Args:
    # weight_matrix: set of perceptron synaptic weights corresponding to each class
    # feature_matrix: set of features (dimensions)
    # error: Difference in expected output and predicted output, calculated using backpropagation
    #Returns:
    # synaptic weights updated in response to backpropagation error
    
    weight_matrix = weight_matrix + learning_rate * error * feature_matrix
    return(weight_matrix)
  }
  
  backpropagate <- function(actual_label, predicted_label, feature_matrix){
    #Function that updates synaptic weights based on error calculated 
    #using delta rule (difference between expected and predicted output)
    #Args:
    # actual_label: known class of the instance
    # feature_matrix: set of features (dimensions)
    # predicted_label: predicted class for given set of inputs

    #Update synaptic weights for predicted class based on the activation function
    label_weights[[predicted_label]] <<- update_synaptic_weights(label_weights[[predicted_label]], -1, feature_matrix)
    #Update synaptic weights for actual class based on the activation function
    label_weights[[actual_label]] <<- update_synaptic_weights(label_weights[[actual_label]], 1, feature_matrix)
  }
  
  list(
    train = function(actual_label, features){
      #Function that trains a perceptron to classify based on given inputs
      #
      #Args:
      # actual_label: known class of the instance
      # features: set of dimensions that are used to predict unknown instances
      #
      #Returns:
      # predicted_label: predicted class for given set of inputs
      #
      #check if weights for training class label exist, add if does not exists
      if (!(actual_label %in% names(label_weights))){
        label_weights[[actual_label]] <<- features
      }

      #Let Perceptron predict the outcome for given the inputs
      predicted_label <- decide_class(features)

      #Back propagate the error in case of incorrect prediction,
      #here the synaptic weights are updated according to contribution of each input
      #so that perceptron "learns" to classify over a period of time (iterations)
    
      if(predicted_label != actual_label){
        backpropagate(actual_label, predicted_label, features)
      }
      predicted_label
    },
    
    predict = function(input_features) {
      #Function that predicts output class value based upon feature values given as inputs
      #
      #Args:
      # feature_set: set of input features for a single instance of data (training/ testing set)
      #
      #Returns:
      # Outputs class label with highest weighted sum 
      #(uses synaptic weights updated during training)
      decide_class(input_features)
      },
     get_learned_weights = function() {
       #Function that can be used to observe the values of weights before/ after or during training
       #
       #Returns:
       # Synaptic weights for different classes
       return(label_weights)
     }
  )
}