from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk import word_tokenize, sent_tokenize

def tokenize_and_stem(paper_text):
  '''
  Tokenizes document & stem words
  Input: string
  Output: list of strings
  '''
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(word) for sentence in sent_tokenize(paper_text) for word in word_tokenize(sentence)]
  return tokens

def create_tfidf(df, column_name, num_features, ngram=1):
    '''
    Creates a sparse matrix of tfidf vectors
    Input: df - Pandas DataFrame
           column_name - String
           num_features - Int
    Output: Scipy Sparse Matrix 
    '''
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,ngram), 
                                       stop_words='english', 
                                       max_df=0.95,
                                       min_df=2,
                                       max_features=num_features,
                                       tokenizer=tokenize_and_stem)
    tfidf_sparse = tfidf_vectorizer.fit_transform(df[column_name]) 
    return tfidf_vectorizer, tfidf_sparse

def create_count(df, column_name, num_features, ngram=1):
    '''
    Creates a sparse matrix of count vectors
    Input: df - Pandas DataFrame 
           column_name - String
           num_features - Int
    Output: Scipy Sparse Matrix 
    '''
    count_vectorizer = CountVectorizer(ngram_range=(1,ngram), 
                                       stop_words='english', 
                                       max_df=0.95,
                                       min_df=2,
                                       max_features=num_features,
                                       tokenizer=tokenize_and_stem)
    count_sparse = count_vectorizer.fit_transform(df[column_name]) 
    return count_vectorizer, count_sparse