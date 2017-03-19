from sklearn.cluster import KMeans

def create_kmeans_model(sparse_matrix, num_clusters):
    '''
    Creates a sklearn kmeans model
    Input: sparse_matrix - Scipy Sparse Matrix 
           num_topics - Int
    Output: model - sklearn kmeans model
            clusters - List (Cluster label index for each sample)
            cluster_space - Numpy Array (Sparse Matrix transformed to cluster-distance space)
    '''
    model = KMeans(n_clusters=num_clusters, n_jobs=-1)
    model.fit(sparse_matrix)
    clusters = model.predict(sparse_matrix)
    cluster_space = model.transform(sparse_matrix)
    return model, clusters, cluster_space