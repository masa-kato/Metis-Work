import numpy as np
from metrics import euclidean, accuracy
import sys
sys.path.append('/Users/Masa/Documents/Data Science/Metis/6. Projects/Project-McNulty/python/mklearn')

class base(object):
    def get_accuracy(self, y_test, y_predicted):
        return accuracy(y_test, y_predicted)    


class KNearestNeighbors_MK(base):
    def __init__(self, k=1, distance=euclidean):
        '''
        Input:
            k - Number of Neighbors
            distance - Distance Metric (euclidean)
        '''
        self.k = k
        self.distance = distance
    
    def fit(self, X_train, y_train):
        '''
        Input:
            X_train - Features of Training Set
            y_train - Target Labels of Training Set
        '''
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        '''
        Input:
            X_test - Features of Test Set
        Output: 
            Numpy Array of Predictions
        '''        
        X_test = np.array(X_test)  
        distances = self.distance(self.X_train, X_test)
        
        self.k_index = np.argsort(distances)[:,:self.k]
        k_targets = self.y_train[self.k_index].astype(np.int64)
        
        predictions = []
        for target in k_targets:
            predictions.append(np.argmax(np.bincount(target)))
        return np.array(predictions)
    
    def get_nearest_neighbors_index(self):
        '''
        Input:
            None
        Output:
            The Index Amongst Y_train of the K Nearest Neighbors 
        '''
        return self.k_index


        
        
        
        
        
        
        
        
        
        
        