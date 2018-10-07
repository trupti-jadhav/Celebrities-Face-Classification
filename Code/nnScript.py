
# coding: utf-8

# In[327]:

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import pickle


# In[303]:

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0 + np.exp(-z))



# In[368]:

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,),dtype=np.int8)
    validation_label_preprocess = np.zeros(shape=(10000,),dtype=np.int8)
    test_label_preprocess = np.zeros(shape=(10000,),dtype=np.int8)
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    train_label = train_label[:,np.newaxis]
        
    full_data = np.array(np.vstack((train_data,validation_data,test_data)))
    #features = full_data.shape[1]
    #print(features)
    #for i in range(0,features):
        #print(full_data[[i])
    #    if ((full_data[:][i] == full_data[0][i])).all():
        #if(full_data[0][i]==0):
     #       deleteIndices += [i];
     #       print("xkjfdsjgh")
      #      full_data = full_data[:][-deleteIndices]
    n = np.all(full_data == full_data[0,:],axis=0)
    m = np.all(full_data == full_data[0,:],axis=0).tolist()
    full_data = full_data[:,~n]
   
    selected_features = list(set(range(0,784)) - set(n))
    pickle.dump((selected_features),open('selected_features.pickle','wb'))
    selected_features = pickle.load(open('selected_features.pickle','rb'))
        
    train_data = full_data[0:len(train_data),:] 
    validation_data = full_data[len(train_data):(len(train_data) + len(validation_data)) ,:]
    test_data = full_data[(len(train_data) + len(validation_data)) : (len(test_data) + len(validation_data) + len(train_data)),:]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

# In[308]:

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    d = train_data.shape[1]
    n = train_data.shape[0]
    m = w1.shape[0]
    k = w2.shape[0]
    
    X = np.concatenate((train_data, np.ones((n, 1))),axis = 1)
    #print(X)
    A = np.dot(X,np.transpose(w1))
    sig_Z = sigmoid(A)
    sig_Z = np.concatenate((sig_Z,np.ones((n,1))),axis = 1)
    #print(sig_Z) #50000,10
    Y = np.dot(sig_Z,np.transpose(w2))
    sig_Y = sigmoid(Y)
    
    #print(sig_Y.shape) #(50000,10)
    
#     out = np.zeros((len(train_data),10),dtype = np.int8) 
#     for i in range(0,len(train_data)):
#         out[i][train_label[out]] = 1 #(50000,10)
        
    out=np.zeros((n,n_class)) #creating an array to store corresponding label values - (50000,10) - Magic
    for x in range (0,n):
        
        out[x][train_label[x]]=1
        
    #obj_val = np.dot(np.transpose(out),np.log(sig_Y)) + np.dot(np.transpose((1 - out)),np.log(1 - sig_Y))
    #obj_val = -obj_val
    
    obj_val_1 = np.multiply(out,(np.log(sig_Y)))
    obj_val_2 = np.multiply((1-out),(np.log(1-sig_Y)))
    obj_val = np.sum(obj_val_1 + obj_val_2)
    obj_val = -obj_val
    
#     #Backpropagation
    gradient_hidden_input= np.zeros((m, d+1))
    gradient_output_hidden  = np.zeros((k, m+1))

      
    #Error from Output layer to Hidden Layer
    y_new = sig_Y * (1 - sig_Y)
    error_output_hidden = (sig_Y - out) * y_new
    #print(error_output_hidden.shape) #(50000,10)
    #print(error_output_hidden)
    
    #Gradient weight calculation for Output to Hidden and Hidden to Input
    gradient_output_hidden = np.dot((np.transpose(error_output_hidden)),sig_Z)
    #print(gradient_output_hidden.shape)
    #print(gradient_output_hidden)
    
    #Error from Hidden to Input
    error_hidden_input1 = (np.dot(error_output_hidden,w2))
    error_hidden_input2 = ((1 - sig_Z) * sig_Z)
    error_hidden_input = error_hidden_input1 * error_hidden_input2
    error_hidden_input_transpose = np.transpose(error_hidden_input)
    
    gradient_hidden_input = np.dot((error_hidden_input_transpose),X)
    gradient_hidden_input= gradient_hidden_input[0:n_hidden,:]
    #print(error_hidden_input.shape) #(50000, 51)
    #print(error_hidden_input)
    
    #A = np.concatenate((train_data, np.ones((n, 1))),axis = 1)
 
    #print(gradient_hidden_input.shape)
    #print(gradient_hidden_input)
    
    	
    gradient_hidden_input = (gradient_hidden_input + (lambdaval * w1)) / n
    gradient_output_hidden = (gradient_output_hidden + (lambdaval * w2)) / n
    
    #print("******************")
    
    obj_val = obj_val + ((lambdaval / 2)  * (np.sum(w1*w1) + np.sum(w2*w2)));
    obj_val = obj_val / n;
    # Unroll gradients
    #print("*******************")

    obj_grad = np.concatenate((gradient_hidden_input.flatten(), gradient_output_hidden.flatten()),0)
    #print(obj_grad)
    #Error from Output layer to Hidden Layer
    y_new = sig_Y * (1 - sig_Y)
    error_output_hidden = (sig_Y - out)   #* y_new
    error_output_hidden_transpose = np.transpose(error_output_hidden)
    gradient_output_hidden = np.dot(error_output_hidden_transpose ,sig_Z)
    #obj_grad = np.array([])        
    return (obj_val,obj_grad)
    
# In[310]:

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    n = data.shape[0]
    d = data.shape[1] # Goes to params, calls params, and gets the new Weights from nnObjectiveFunction and runs the Predict function
    
    X = np.concatenate((data, np.ones((n, 1))),1)
    A = np.dot(X,np.transpose(w1))
    sig_Z = sigmoid(A)
    sig_Z = np.concatenate((sig_Z,np.ones((n,1))),axis = 1)
    #print(sig_Z) #50000,10
    Y = np.dot(sig_Z,np.transpose(w2))
    sig_Y = sigmoid(Y)
    
    labels = np.argmax(sig_Y,axis = 1) #Maximum probability value taken for prediction
    labels = labels[:, np.newaxis]
    
    return labels



# In[369]:

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


# In[312]:

import time
column_names = ['lambda','Hidden','Time','Train_Accuracy','Validation_Accuracy','Test_Accuracy']
results = pd.DataFrame(columns = column_names).astype({'lambda' : 'int8','Hidden' : 'int8'})

lambdaval = [0, 10, 20, 30, 40, 50,60]
n_hidden_list = [4, 8, 12, 16, 20]
for i in lambdaval:
    for j in n_hidden_list:
        
        s = time.time()
    # set the number of nodes in input unit (not including bias unit)
        n_input = train_data.shape[1]; 

        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = j;

        # set the number of nodes in output unit
        n_class = 10;				   

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

        # set the regularization hyper-parameter
        lambdaval = i;


        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter' : 50}    # Preferred value.



        #objv, obgg = nnObjFunction(initialWeights, args)

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
        params = nn_params.get('x')
        #In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        #and nnObjGradient. Check documentation for this function before you proceed.
        #nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        #Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        predicted_label = nnPredict(w1,w2,train_data)
        #print("lambda",i)
        #print("Hidden Layer", j)
        

        #find the accuracy on Training Dataset

        print('\n Training set Accuracy:' + str(100*np.mean(predicted_label==train_label.reshape(train_label.shape[0],1))) + '%')
        train_accuracy = 100*np.mean(predicted_label==train_label.reshape(train_label.shape[0],1))

        predicted_label = nnPredict(w1,w2,validation_data)

        #find the accuracy on Validation Dataset

        print('\n Validation set Accuracy:' + str(100*np.mean(predicted_label==validation_label.reshape(validation_label.shape[0],1))) + '%')
        validation_accuracy = 100*np.mean(predicted_label==validation_label.reshape(validation_label.shape[0],1))


        predicted_label = nnPredict(w1,w2,test_data)

        #find the accuracy on Validation Dataset

        print('\n Test set Accuracy:' +  str(100*np.mean(predicted_label==test_label.reshape(test_label.shape[0],1))) + '%')
        test_accuracy = 100*np.mean(predicted_label==test_label.reshape(test_label.shape[0],1))
        
        e=time.time()
        Time = e-s

        
#         idx = results['lambda'].index
#         print(results.set_value(idx, 'lambda', i))
        results = results.append({'lambda':i,'Hidden':j,'Time':Time, 'Train_Accuracy':train_accuracy,'Validation_Accuracy':validation_accuracy,'Test_Accuracy':test_accuracy}, ignore_index=True)
    print(results)
        


# In[313]:

lambdaval_values = results['lambda']
lambda_time = results['Time']

plt.scatter(lambdaval_values,lambda_time)
plt.show()


# In[314]:

hidden_values = results['Hidden']
hidden_time = results['Time']

plt.scatter(hidden_values,hidden_time)
plt.show()


# In[323]:

lambda_mean = results.groupby(results['lambda']).agg({"Time":np.mean })
print(lambda_mean)
lambda_mean.plot.bar()
plt.show()


# In[316]:

hidden_mean = results.groupby("Hidden").agg({"Time":np.mean })
print(hidden_mean)
hidden_mean.plot.bar()
plt.show()


# In[317]:

test_accuracy_mean = results.groupby("lambda").agg({"Test_Accuracy" : np.mean})
print(test_accuracy_mean)
test_accuracy_mean.plot()
plt.show()


# In[318]:

accuracy_mean = results.groupby("lambda").agg({"Train_Accuracy" : np.mean, "Validation_Accuracy" : np.mean, "Test_Accuracy" : np.mean})
print(accuracy_mean)
accuracy_mean.plot()
plt.show()


# In[319]:

hidden_accuracy_mean = results.groupby("Hidden").agg({"Train_Accuracy" : np.mean, "Validation_Accuracy" : np.mean, "Test_Accuracy" : np.mean})
print(hidden_accuracy_mean)
hidden_accuracy_mean.plot()
plt.show()


# In[321]:

hidden_accuracy_mean


# In[322]:

accuracy_mean


# In[372]:

pickle.dump((n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
param_pickle = pickle.load(open('params.pickle','rb'))
print(param_pickle)

