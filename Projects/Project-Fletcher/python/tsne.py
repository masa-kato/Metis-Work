from sklearn.manifold import TSNE
import pandas as pd

def create_tsne_model(topic_space_matrix, dimensions, random_state=0, metric='euclidean', perplexity=30):
    '''
    Creates t-SNE model 
    Input: topic_space_matrix - Numpy Array
           dimensions - Int (2 or 3)
           random_state - Int
           metric - Str (Sklearn pairwise distance metric)
    Output: Sklearn t-SNE Model
            Numpy Array (Embedding of the training data in low-dimensional space)
    '''
    model = TSNE(n_components=dimensions, metric=metric, random_state=random_state, perplexity=perplexity)
    tsne_matrix = model.fit_transform(topic_space_matrix)
    return model, tsne_matrix

def create_tsne_df_2D(tsne_matrix, topic_space_matrix, df):
    '''
    Creates a Pandas DataFrame with Columns X, Y, Year, & Topic
    Input: tsne_matrix - Numpy Array
           topic_space_matrix - Numpy Array
           df - Pandas DataFrame (Original DataFrame)
    Output: Pandas DataFrame
    '''
    # Put matrix into a Pandas DataFrame
    tsne_df = pd.DataFrame(tsne_matrix, columns=['X','Y'])
    tsne_df['Year'] = df['year']
    tsne_df['Topic'] = topic_space_matrix.argmax(axis=1)    
    return tsne_df

def create_tsne_df_3D(tsne_matrix, topic_space_matrix, df):
    '''
    Creates a Pandas DataFrame with Columns X, Y, Year, & Topic
    Input: tsne_matrix - Numpy Array
           topic_space_matrix - Numpy Array
           df - Pandas DataFrame (Original DataFrame)
    Output: Pandas DataFrame
    '''
    # Put matrix into a Pandas DataFrame
    tsne_df = pd.DataFrame(tsne_matrix, columns=['X','Y','Z'])
    tsne_df['Year'] = df['year']
    tsne_df['Topic'] = topic_space_matrix.argmax(axis=1)    
    return tsne_df

