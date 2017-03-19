# Import Data & Create DataFrame
import pandas as pd
import numpy as np

def unpickle(filepath):
    '''
    Unpickles Data Files & Creates Pandas DataFrame
    
    Input:
        filepath: File path of cifar-10 data
    Output:
        Pandas DataFrame
    '''
    dic = pd.read_pickle(filepath)
    features = pd.DataFrame(dic['data'])
    labels = pd.DataFrame({'Labels':dic['labels']})
    df = pd.concat([labels,features],axis=1)
    return df

def create_dataframe():
    '''
    Combines Data Batch into A Single Pandas DataFrame
    
    Input: 
        None
    Output:
        Pandas DataFrame
    '''
    batch_1 = unpickle('../data/cifar-10/data_batch_1')
    batch_2 = unpickle('../data/cifar-10/data_batch_2')
    batch_3 = unpickle('../data/cifar-10/data_batch_3')
    batch_4 = unpickle('../data/cifar-10/data_batch_4')
    batch_5 = unpickle('../data/cifar-10/data_batch_5')
    df = pd.concat([batch_1,batch_2,batch_3,batch_4,batch_5],axis=0).reset_index(drop=True)
    return df

def create_rbf_array(df):
    '''
    Creates Numpy Array of RBF Pixels
    
    Input:
        Pandas DataFrame of Entire Train Set (50000 Samples)
    Output: 
        Numpy Array of RBF Pixels
    '''
    matrix = np.array(df.iloc[:,1:]).reshape(50000,3,32,32).transpose(0,2,3,1)
    return matrix

def create_rbf_array_sorted(df):
    '''
    Creates Numpy Array of RBF Pixels Sorted by Class
    
    Input:
        Pandas DataFrame of Entire Train Set (50000 Samples)
    Output: 
        Numpy Array of RBF Pixels Sorted by Class
    '''
    df = df.sort_values(by='Labels').reset_index(drop=True)
    matrix = np.array(df.iloc[:,1:]).reshape(50000,3,32,32).transpose(0,2,3,1)
    return matrix