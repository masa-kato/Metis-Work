from sklearn.decomposition import NMF

def create_nmf_model(sparse_matrix, num_topics, lambda_reg=0.1, l1_ratio=0.5):
    '''
    Creates an NMF model
    Input: sparse_matrix - Scipy Sparse Matrix 
           num_topics - Int 
           lambda_reg - Float; Regularization Term
    Output: nmf - Sklearn NMF model
            nmf_topicspace - Numpy Array (Sparse matrix transformed into topic space)
    '''
    nmf = NMF(n_components=num_topics, 
              alpha=lambda_reg, 
              l1_ratio=l1_ratio,
              random_state=0)
    nmf.fit(sparse_matrix)
    nmf_topicspace = nmf.transform(sparse_matrix)
    return nmf, nmf_topicspace

def print_topics(vectorizer, nmf_model, num_words):
    '''
    Prints the top words for each topic created by the nmf model
    Input: vectorizer - Sklearn Vectorizer 
           nmf_model - Sklearn NMF model
           num_words - Int     
    Output: None
    '''
    word_list = vectorizer.get_feature_names()
    components = nmf_model.components_
    for i in range(len(components)):
        top_words_index = components[i].argsort()[::-1][:num_words]
        top_words = [word_list[index] for index in top_words_index]
        
        print('Topic {}'.format(i+1))
        print(top_words)

def print_top_documents_per_topic(nmf_topicspace, num_documents, df):
    '''
    Prints the top documents for each topic created by the nmf model
    Input: nmf_topicspace - Numpy Array (Matrix of tfidf vector transformed to topic space)
           num_documents - Int (Number of documents to show per topic)   
           df - Pandas DataFrame
    Output: None    
    '''
    index_per_topic = nmf_topicspace.transpose().argsort(axis=1)
    
    for i, topic in enumerate(index_per_topic):
        top_indices = topic[-num_documents:][::-1]
        print('\nTopic {}'.format(i+1))
        for index in top_indices:
            title = df['title'][index]
            score = nmf_topicspace[index].max()
            print('{}, Score: {:.4f}'.format(title, score))