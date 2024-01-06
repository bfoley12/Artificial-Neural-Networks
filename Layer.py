class Layer:
   def __init__(self, input, output):
      self.input = None
      self.output = None
   
   def forward_propagation(self, input):
      raise NotImplementedError
   
   def backward_propagation(self, error, learning_rate):
      raise NotImplementedError
