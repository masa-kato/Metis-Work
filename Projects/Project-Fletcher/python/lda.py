from sklearn.decomposition import LatentDirichletAllocation

def create_lda_model(sparse_matrix, num_topics):
    '''
    Creates a sklearn latent dirichlet allocation model
    Input: sparse_matrix - Scipy Sparse Matrix 
           num_topics - Int
    Output: lda - sklearn lda model
            lda_topicspace - Numpy Array (Sparse matrix transformed into topic space)
    '''
    
    lda = LatentDirichletAllocation(n_topics=num_topics, 
                                    random_state=0,
                                    learning_method='batch')
    lda_topicspace = lda.fit_transform(sparse_matrix)
    return lda, lda_topicspace