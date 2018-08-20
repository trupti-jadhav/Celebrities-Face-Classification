
# coding: utf-8

# In[35]:

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0
    #print(train_label)
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    #print(train_data.shape)    #5000
    #print(labeli.shape)    #715
    n_data = train_data.shape[0]    
    n_features = train_data.shape[1]    
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.concatenate((np.ones((n_data,1)), train_data),axis =1) # bias added 50000,716
    #w_transpose = np.transpose(initialWeights) #1,716
    initialWeights = initialWeights[:,np.newaxis]
    #print(initialWeights.shape) #
    theta = np.dot(X,initialWeights)
    sig = sigmoid(theta)
    #print(sig.shape) # (50000, 1)
    
    # calculating error
    error1 = labeli*np.log(sig)
    error2 = (1.0-labeli)*np.log(1.0-sig)
    error = np.sum(error1 + error2)
    error = -(error/n_data)
    
    #calculate error gradient
    error_grad = (sig - labeli)*X
    error_grad = np.sum(error_grad, axis =0)/n_data
    #print(error.shape)
    #print(error_grad.shape)

    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix
                    
    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.concatenate((np.ones((n_data,1)), data),axis =1)
    theta = np.dot(X,W)
    # calculate posterior probability
    posterior = sigmoid(theta)
    #print(posterior.shape)
    label = np.argmax(posterior, axis=1)
    label = label[:,np.newaxis]

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()
#print(Y.shape) #50000,10
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class)) # adding bias 716*10
initialWeights = np.zeros((n_feature + 1, 1)) #setting weights as 0
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1) #(50000, 1)
    #print(labeli.shape)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

    
# Find the accuracy on Training Dataset
from sklearn.metrics import classification_report
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_train = blrPredict(W, train_data)
train_conf = confusion_matrix(train_label, predicted_train, labels=None, sample_weight=None)
print(train_conf)
print(classification_report(train_label, predicted_train, labels=None, target_names=None, sample_weight=None, digits=2))

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
predicted_test = blrPredict(W, test_data)
#test_conf = confusion_matrix(test_label, predicted_test, labels=None, sample_weight=None)
print(test_conf)
print(classification_report(test_label, predicted_test, labels=None, target_names=None, sample_weight=None, digits=2))

def softmax_transform(theta):
    exp = np.exp(theta)
    #print(exp.shape)
    n = np.sum(exp, axis = 1)
    n = n.reshape(exp.shape[0],1)
    #print(n.shape)
    posterior = exp/n
    #rint(posterior)
    return posterior

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    initialWeights=params.reshape(716,10)
    train_data, Y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.concatenate((np.ones((n_data,1)), train_data),axis =1) # bias added 50000,716
    initialWeights = initialWeights.reshape(n_feature+1,10)  # 716,10
    theta = np.dot(X,initialWeights)
    sig = softmax_transform(theta)
    #print(sig.shape) # (50000, 10)
    #print(labeli.shape) # (50000, 10)
    
    # calculating error
    error1 = np.sum(np.sum(Y*np.log(sig)))
    error = -1.0*error1
    print(error)
    
    
    #calculate error gradient
    error_grad = np.dot(np.transpose(X),(sig - Y))
    #print("working")
    #error_grad = np.transpose(error_grad)
    #print(error_grad.shape)
    error_grad = error_grad.ravel()
    #print(error_grad.shape)
    



    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.concatenate((np.ones((n_data,1)), data),axis =1)
    W = W.reshape(n_feature+1,10)
    theta = np.dot(X,W)
    # calculate posterior probability
    posterior = softmax_transform(theta)
    #print(posterior.shape)
    label = np.argmax(posterior, axis=1)
    label = label[:,np.newaxis]

    return label

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
train_label_ravel = np.ravel(train_label)

print("SVM Code Starting Here")
print("\n")
print("SVM with Linear Kernel")

train_label_ravel = np.ravel(train_label)

clf = SVC(kernel = 'linear')
clf.fit(train_data, train_label_ravel)
train_score = clf.score(train_data, train_label, sample_weight=None)
validation_score = clf.score(validation_data, validation_label, sample_weight=None)
test_score = clf.score(test_data, test_label, sample_weight=None)

print('\n Training set Accuracy:' + str(100 * train_score) + '%')
print('\n Validation set Accuracy:' + str(100 * validation_score) + '%')
print('\n Test set Accuracy:' + str(100 * test_score) + '%')

print("SVM with Radial Basis Function Kernel and Gamma = 1")

clf = SVC(kernel = 'rbf',gamma = 1)
clf.fit(train_data, train_label_ravel)
train_score = clf.score(train_data, train_label, sample_weight=None)
validation_score = clf.score(validation_data, validation_label, sample_weight=None)
test_score = clf.score(test_data, test_label, sample_weight=None)

print('\n Training set Accuracy:' + str(100 * train_score) + '%')
print('\n Validation set Accuracy:' + str(100 * validation_score) + '%')
print('\n Test set Accuracy:' + str(100 * test_score) + '%')

print("SVM with Radial Basis Function Kernel")

clf = SVC(kernel = 'rbf')
clf.fit(train_data, train_label_ravel)
train_score = clf.score(train_data, train_label, sample_weight=None)
validation_score = clf.score(validation_data, validation_label, sample_weight=None)
test_score = clf.score(test_data, test_label, sample_weight=None)

print('\n Training set Accuracy:' + str(100 * train_score) + '%')
print('\n Validation set Accuracy:' + str(100 * validation_score) + '%')
print('\n Test set Accuracy:' + str(100 * test_score) + '%')

print("SVM with Radial Basis Function Kernel")

C = [1,10,20,30,40,50,60,70,80,90,100]
test_score_list = []
train_score_list =[]
validation_score_list = []
for i in C:
    #print(i)
    clf = SVC(kernel = 'rbf', C = i)
    clf.fit(train_data[1:10000,], train_label_ravel[1:10000, ])
    train_score = clf.score(train_data, train_label, sample_weight=None)
    validation_score = clf.score(validation_data, validation_label, sample_weight=None)
    test_score = clf.score(test_data, test_label, sample_weight=None)
    test_score_list.append(test_score*100)
    train_score_list.append(train_score*100)
    validation_score_list.append(validation_score*100)
    #print(test_score_list)

print('\n Training set Accuracy:' + str(100 * train_score) + '%')
print('\n Validation set Accuracy:' + str(100 * validation_score) + '%')
print('\n Test set Accuracy:' + str(100 * test_score) + '%')
print("Test Accuracy for each value of C", test_score_list)
print("Train Accuracy for each value of C", train_score_list)
print("Validation Accuracy for each value of C", validation_score_list)
plt.plot(C,test_score_list, label = "Test Accuracy")
plt.plot(C,train_score_list, label = "Train Accuracy")
plt.plot(C,validation_score_list, label = "Validation Accuracy")
plt.legend()
plt.show()