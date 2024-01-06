import numpy as np
from Layer import Layer
class HiddenLayer(Layer):
   def __init__(self, input, output):
      """Creates an instance

      Args:
         input (int): number of inputs
         output (int): number of outputs
      """
      self.weights = np.random.uniform(-0.1, 0.1, (input, output))
      self.bias = 1

   def forward_propagation(self, input):
      """Handles foward propagation
      
      Args:
         input (np.array): inputs coming from previous layer
      
      Return
         self.output (np.array): output of the layer
      """
      self.input = input
      self.output = np.dot(self.input, self.weights) + self.bias
      return self.output
   
   def backward_propagation(self, error, learning_rate):
      """Handles correction of weights
      
      Args:
         error (np.array): error of the weight
         learning_rate (float): rate to apply correction
      
      Return:
         input_error (np.array): error of inputs to be backpropagated
      """
      input_error = np.dot(error, self.weights.T)
      weights_error = np.dot(self.input.T, error)
      # dBias = output_error

      # update parameters
      self.weights -= learning_rate * weights_error
      self.bias -= learning_rate * error
      return input_error
