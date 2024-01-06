import numpy as np
import pandas as pd
import Toolkit
from Network import Network
from HiddenLayer import HiddenLayer
from ActivationLayer import ActivationLayer
import csv

def mse(actual, pred):
    """Mean Square Error calculation
    
    Args:
      actual (float array): expected values
      pred (float array): predicted values
   
   Return:
      The mean square error of the two arrays
   """
    return np.mean(np.power(actual - pred, 2))

def mse_prime(actual, pred):
    """Derivative of Mean Square error
    
    Args:
      actual (float array): expected values
      pred (float array): predicted values
      
   Return:
      The derivative of mean square error between two arrays
   """
    return 2 * (pred-actual) / actual.size

def cross_entropy(y_true, y_pred):
    """Cross entropy calculation
    
    Args:
      y_true (float array): expected values
      y_pred (float array): predicted values
      
   Return:
      The cross entropy between the two arrays
   """
    m = len(y_true)
    p = y_pred
    log_likelihood = -np.log(p[0, np.argmax(y_true)] + 1e-15)
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_prime(y_true, y_pred):
    """Derivative of cross entropy calculation
    
    Args:
      y_true (float array): expected values
      y_pred (float array): predicted values
      
   Return:
      The derivative of the cross entropy between the two arrays
   """
    m = len(y_true)
    grad = y_pred - y_true
    return grad/m

def tanh(x):
   """Hyperbolic Tangent activation function
   
   Args:
      x (float array): input to calculate tanh on
      
   Return:
      the tanh of the input
   """
   return np.tanh(x)
def tanh_prime(x):
   """Derivative of the hyperbolic tangent
   
   Args:
      x (float array): input to calculate tanh on
      
   Return:
      the derivative of tanh of the input
   """
   return 1-np.tanh(x)**2

def linear(x):
    """Linear activation function
    
    Args:
      x (float): input to activate
   
   Return:
      x: the input
   """
    return x
def linear_prime(x):
    """Derivative of the linear function
    
    Args:
      x (float): input to take the derivative of
   
   Returns:
      1 : the derivative of x
   """
    return 1

def softmax_logits(logits):
    """Softmax activation function with logits
    
    Args:
      logits (float array): input to take softmax of
      
   Return:
      softamx of input
   """
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def softmax_logits_prime(logits):
    """Derivative of the softmax
    
    Args: 
      logits (float array): input to take derivative of
      
   Return:
      the derivative of the input
   """
    p = softmax(logits)
    return p * (1 - p)

def softmax(x):
   """Softmax activation function with logits
    
    Args:
      logits (float array): input to take softmax of
      
    Return:
      softamx of input
   """
   return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def softmax_prime(x):
   """Derivative of the softmax
    
    Args: 
      logits (float array): input to take derivative of
      
   Return:
      the derivative of the input
   """
   return softmax(x) * (1 - softmax(x))

def accuracy(actual, pred):
   """Accuracy between actual and predicted outputs
   
   Args:
      actual (float array): expected values
      pred (float array): predicted values
      
   Return:
      The accuracy between the two arrays
   """
   correct = 0
   for i in range(len(pred)):
         if actual[i][pred[i]] == 1:
            correct += 1
   return correct / len(pred)

def one_hot_encode(labels):
    """One-hot encodes class labels
    
    Args:
      labels (array): the labels to one hot encode
      
   Return:
      one_hot(array): the class labels encoded as a one-hot vector
   """
    unique_labels = np.unique(labels)
    num_labels = len(labels)
    num_classes = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[label] for label in labels])
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), mapped_labels] = 1
    return one_hot


def write_2dcsv(filename, arr):
   """Writes 2d array to csv
   Helper method for analysis and writing
   
   Args:
      filename (string): filename (plus extension) to write to
      arr (2d array): 2d array we wish to write
   """
   with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in arr:
                writer.writerow(row)

def generate_sets(data):
   """Partitions data into training, test, and validation sets
   
   Args:
      data (pd.dataframe): dataframe to be partitioned
      
   Return:
      training (pd.dataframe): training data
      test (pd.dataframe): test data
      validation (pd.dataframe): validation data
   """
   validation = data.groupby('class', group_keys = False).apply(lambda g: g.sample(frac=0.2))
   for i in range(5):
      training = data.loc[~data.index.isin(validation.index)]
      test = training.groupby('class', group_keys = False).apply(lambda g: g.sample(frac = .5))
      training = training.loc[~training.index.isin(test.index)].reset_index(drop=True)
      test = test.reset_index(drop = True)
      validation = validation.reset_index(drop = True)
      

      for column in training:
         if training[column].dtype != 'category' and column != 'class':
            m = training[column].mean()
            s = training[column].std()
            training[column] = (training[column] - m) / s
            test[column] = (test[column] - m) / s
         elif column != 'class':
            training[column] = training.groupby([column, 'class'])[column].transform(lambda x : x.count()/len(training))
            training[column] = pd.to_numeric(training[column])        
            test[column] = test.groupby([column, 'class'])[column].transform(lambda x : x.count()/len(test))
            test[column] = pd.to_numeric(test[column])    
   return training, test, validation

def run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, autoencoder):
   """Runs an ANN
   
   Args:
      net (Network): network to run
      training (pd.dataframe): data to train with
      y_training (np.array): vector of actual labels/outputs
      test (pd.dataframe): data to test with
      y_test (np.array): vector of actual labels/outputs
      epochs (int): number of epochs
      learning_rate (float): rate to learn
      classification (booleanr): classification task flag
   
   Return:
      acc (float): accuracy of network
   """
   simple_errors = net.train(training, y_training, epochs, learning_rate, autoencoder)
   res = net.predict(test)

   predictions = []
   output = []
   acc = []
   i=0
   for r in res:
      if classification:
         output.append(softmax(r)[0])
         predictions.append(np.argmax(output[i]))
      else:
            output.append(linear(r)[0][0])
      i+=1
   if classification:
         acc.append(accuracy(y_test, predictions))
   else:
      acc.append(mse(y_test,output))

   return acc
def column_mean(col):
   return sum(col) / len(col)

for opt in range(1, 7):
   epochs = 50
   simple_accuracy = [[], [], [], [], []]
   simple_jk = [[], [], [], [], []]
   two_layer_accuracy = [[], [], [], [], []]
   two_layer_jk = [[], [], [], [], []]
   autoencoder_accuracy = [[], [], [], [], []]
   autoencoder_jk = [[], [], [], [], []]
   for i in range(5):
      print(i+1)
      match opt:
         case 1:
            learning_rate = 0.1
            classification = 1
            print("Training Class 1") 

            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            data = data.drop('id', axis=1)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)[0])
                  autoencoder_jk[i].append([j, k])

            #with open('Case1_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            
            #write_2dcsv("Case1_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case1_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case1_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case1_Final_Autoencoder_Errors.csv", final_autoencoder_errors)
         case 2:
            learning_rate = 0.1
            classification = 1
            print("Training Class 2")
            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)[0])
                  autoencoder_jk[i].append([j, k])
            #with open('Case2_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            #write_2dcsv("Case2_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case2_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case2_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case2_Final_Autoencoder_Errors.csv", final_autoencoder_errors)

         case 3:
            learning_rate = 0.1
            classification = 1
            print("Training Class 3")
            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)[0])
                  autoencoder_jk[i].append([j, k])
            #with open('Case3_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            
            #write_2dcsv("Case3_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case3_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case3_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case3_Final_Autoencoder_Errors.csv", final_autoencoder_errors)
         case 4:
            learning_rate = 0.1
            classification = 1
            print("Training Class 4")
            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0]), 1))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)[0])
                  autoencoder_jk[i].append([j, k])
            
            #with open('Case4_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            #write_2dcsv("Case4_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case4_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case4_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case4_Final_Autoencoder_Errors.csv", final_autoencoder_errors)
         case 5:
            learning_rate = 0.1
            classification = 1
            print("Training Class 5")
            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, 1)[0])
                  autoencoder_jk[i].append([j, k])
            
            #with open('Case5_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            #write_2dcsv("Case5_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case5_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case5_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case5_Final_Autoencoder_Errors.csv", final_autoencoder_errors)
         case 6:
            learning_rate = 0.1
            classification = 1
            print("Training Class 6")
            # SAMPLE PREP
            data = Toolkit.load_from_csv(opt)
            training, test, validation = generate_sets(data)

            training['class'] = one_hot_encode(training['class'].values).tolist()
            test['class'] = one_hot_encode(test['class'].values).tolist()
            y_training = training['class']
            training = training.drop('class', axis = 1)

            y_test = test['class']
            test = test.drop('class', axis = 1)
            # SIMPLE NETWORK
            simple_net = Network()
            simple_net.add_layer(HiddenLayer(len(training.columns), len(y_training[0])))
            simple_net.add_layer(ActivationLayer(softmax, softmax_prime))
            simple_net.set_loss(cross_entropy, cross_entropy_prime)
            simple_accuracy[i].append(run_net(simple_net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])

            # TWO LAYER NETWORK
            for j in range(4, len(training.columns) + 1):
               for k in range(4, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(j, k))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(k, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  two_layer_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 0)[0])
                  two_layer_jk[i].append([j, k])

            # AUTOENCODER NETWORK
            for j in range(1, len(training.columns)):
               for k in range(1, j):
                  net = Network()
                  net.add_layer(HiddenLayer(len(training.columns), len(training.columns)-j))
                  net.add_layer(ActivationLayer(tanh, tanh_prime))
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  net.set_loss(cross_entropy, cross_entropy_prime)
                  run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)
                  net.pop_layer()
                  net.pop_layer()
                  net.add_layer(HiddenLayer(len(training.columns)-j, len(y_training[0])))
                  net.add_layer(ActivationLayer(softmax_logits, softmax_logits_prime))
                  autoencoder_accuracy[i].append(run_net(net, training, y_training, test, y_test, epochs, learning_rate, classification, 1)[0])
                  autoencoder_jk[i].append([j, k])
            
            #with open('Case6_Accuracies.csv', 'w', newline = '') as csvfile:
            #   writer = csv.writer(csvfile)
            #   writer.writerow(simple_accuracy)
            #   writer.writerow(two_layer_accuracy)
            #   writer.writerow(autoencoder_accuracy)
            #write_2dcsv("Case6_Simple_errors.csv", simple_errors)
            #write_2dcsv("Case6_Two_Layer_Errors.csv", two_layer_errors)
            #write_2dcsv("Case6_Autoencoder_Errors.csv", autoencoder_errors)
            #write_2dcsv("Case6_Final_Autoencoder_Errors.csv", final_autoencoder_errors)
   best_two_layer = np.argmax(np.mean(two_layer_accuracy, axis = 0))
   best_autoencoder = np.argmax(np.mean(autoencoder_accuracy, axis = 0))
   print("Case " + str(opt) + " best 2 layer: " + str(best_two_layer))
   print("j = " + str(two_layer_jk[0][best_two_layer][0]))
   print("k = " + str(two_layer_jk[0][best_two_layer][1]))
   print("Case " + str(opt) + " best autoencoder: " + str(best_autoencoder))
   print("j = " + str(autoencoder_jk[0][best_autoencoder][0]))
   print("k = " + str(autoencoder_jk[0][best_autoencoder][1]))
print()
