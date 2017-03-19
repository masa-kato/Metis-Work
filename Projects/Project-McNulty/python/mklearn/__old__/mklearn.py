import pandas as pd
import numpy as np
from metrics import euclidean, L1
from collections import Counter

import sys
sys.path.append('/Users/Masa/Documents/Data Science/Metis/6. Projects/Project-McNulty/python/mklearn')


class KNearestNeighbors_MK(object):
    def __init__(self, k=1, distance=euclidean):
        '''
        Input:
            k - Number of Neighbors
            distance - Distance Metric (euclidean, L1)
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
            Numpy Array of Index of K Nearest Neighbors Amongst X_train
        '''
        X_test = np.array(X_test)
    
        predictions = []
        k_indices = []
        for x in X_test:
            prediction, k_index = self.predict_single(x)
            predictions.append(prediction)
            k_indices.append(k_index)
        
        return np.array(predictions), np.array(k_indices)
        
    
    def predict_single(self, x):
        '''
        Input:
            x - Features of Single Observation 
        Output:
            prediction - Single Prediction Value
            k_index - Index of K Nearest Neighbors Amongst X_train
        '''
        x = np.array(x)
        
        distances = self.distance(self.X_train, x)
        
        index = range(len(self.y_train))
        labels = list(zip(distances, self.y_train, index))
        
        nearest = sorted(labels, key=lambda x:x[0])
        
        k_nearest= nearest[:self.k]
        k_index = [tup[-1] for tup in k_nearest]
        k_targets = [tup[1] for tup in k_nearest]

        # Predict Most Common Value in K Nearest Neighbors
        count = Counter(k_targets)
        prediction = count.most_common()[0][0]

        return prediction, np.array(k_index)
        
    def predict_x(self, X_test):
        '''
        Input:
            X_test - Features of Test Set
        Output: 
            Numpy Array of Predictions
        '''        
        X_test = np.array(X_test)
        
        distances = self.distance(self.X_train, X_test)
        
        # Index of K Nearest Neighbors
        k_index = np.argsort(distances)[:,:self.k]
        self.k_index = k_index
        
        # Lables of K Nearest Neighbors
        k_targets = self.y_train[k_index].astype(np.int64)
        
        predictions = []
        for target in k_targets:
            predictions.append(np.argmax(np.bincount(target)))
        
        return np.array(predictions)
    
    def get_nearest_neighbors_index(self):
        return self.k_index
        
        
        
        
        
        
        
        
        
        
        
        