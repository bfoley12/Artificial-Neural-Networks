import numpy as np
class Network:
   def __init__(self):
      """Creates an instance"""
      self.layers = []
      self.loss = None
      self.loss_prime = None

   def add_layer(self, layer):
      """Adds a layer to network
      
      Args:
         layer (Layer): the layer to be added
      """
      self.layers.append(layer)

   def pop_layer(self):
       """Removes most recently added layer"""
       self.layers.pop()

   def set_loss(self, loss, loss_prime):
      """Sets the loss function to use for the network
      
      Args: 
         loss (function): loss function to use
         loss_prime (function): derivative of loss function used
      """
      self.loss = loss
      self.loss_prime = loss_prime
   
   def predict(self, input):
      """Predicts output of inputs
      
      Args:
         input (pd.dataframe): the inputs to predict
         
      Return:
         res (np.array): the output predictions"""
      res = []

      for i in range(len(input)):
         output = input.iloc[i]
         for layer in self.layers:
            output = layer.forward_propagation(output)
         res.append(output)

      return res
   
   def train(self, x, y, epochs, learning_rate, autoencoder = 0):
      """Trains the network
      
      Args:
         x (pd.dataframe): data to train with
         y (np.array): correct outputs
         epochs (int): number of epochs
         learning_rate (float): rate to learn
         autoencoder (boolean): whether the network uses an autoencoder
      
      Return:
         ret_error (np.array): error of every epoch - used for analysis
      """
      samples = len(x) 
      ret_error = []

      for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x.iloc[j].to_numpy().reshape((1, len(x.iloc[j])))
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                # CHECK OTHER IMPLEMENTATIONS TO SEE IF y.iloc[j] CHANGES IMPLEMENTATION FROM y[j]

                err += self.loss(y[j], output)

                # backward propagation
                
                error = self.loss_prime(y[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            ret_error.append(err)
            #print('epoch %d/%d   error=%f' % (i+1, epochs, err))
      return ret_error