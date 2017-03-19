import numpy as np

def euclidean(X_train, X_test):
    '''
    Input:
        X_train - Features of Training Set
        X_test - Features of Test Set
    Output:
        Array of Distances for Each Observation in X_train
    '''
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    
    if len(X_test) == 1:
        dist = np.sqrt(np.sum(np.square(X_train - X_test),axis=1))
        return dist
        
    else:
        dist = np.sqrt(np.sum(np.square(X_train),axis=1,keepdims=True) + \
                       np.sum(np.square(X_test),axis=1) - \
                       2*np.dot(X_train, np.transpose(X_test)))
        return np.transpose(dist)
        
        
def L1(X_train, x):
    '''
    Input:
        X_train - Features of Training Set
        x - Features of Single Observation 
    Output:
        Array of Distances for Each Observation in X_train
    '''
    return np.sum(np.abs(X_train - x, dtype=np.float64),axis=1)
    

def accuracy(y_test, y_predictions):
    '''
    Input:
        y_test - Targets in Test Set
        y_predictions - Predicted Targets
    Output:
        Single Value for Accuracy
    '''
    return sum(y_test == y_predictions)/len(y_test)

