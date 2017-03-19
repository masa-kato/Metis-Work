
# Data Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Image Plots
def visualize_random(matrix):
    '''
    Plots 100 Random Images from Dataset
    
    Input: 
        Numpy Array of RBF Pixels
    Output:
        Image Plots
    '''
    random_indices = np.random.choice(50000, 100)
    
    plt.figure(figsize=(20,20))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(matrix[random_indices[i]])
        plt.axis('off')    

def visualize_class(matrix, classes):
    '''
    Plots 10 Random Images per Class from Dataset
    
    Input: 
        matrix - Numpy Array of RBF Pixels
        classes - Name of Classes
    Output:
        Image Plots
    '''
    random_indices = np.random.choice(range(5000),10)
    
    plt.figure(figsize=(20,20))
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i*10+j+1)
            plt.imshow(matrix[random_indices[i] + 5000*j])
            plt.axis('off')    
            if i == 0:
                plt.title(classes[j], fontsize=16)

def visualize_neighbors(rbf_sorted, nearest_neighbors):
    '''
    Plots Images of Nearest Neighbors
    
    Input:
        rbf_sorted - Numpy Array of RBF Pixels Sorted by Class
        nearest_neighbors - Numpy Array of Index of Images 
    Output:
        Image Plots
    '''
    x, y = nearest_neighbors.shape

    plt.figure(figsize=(20,20))
    for i in range(x):
        observation = nearest_neighbors[i]
        for j in range(y):
            image_index = observation[j]
            plt.subplot(x, y, 1+j + y*i)
            plt.imshow(rbf_sorted[image_index])
            plt.axis('off')
            if j == 0:
                plt.title('Input Image', fontsize=16)