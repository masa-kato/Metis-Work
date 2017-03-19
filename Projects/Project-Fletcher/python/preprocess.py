def import_raw_data():
    '''
    Imports data from sqlite file
    Input: None
    Output: Pandas DataFrame
    '''
    import sqlite3
    cnx = sqlite3.connect('../data/nips-papers/database.sqlite')
    df = pd.read_sql_query('''
    SELECT papers.id AS id, year, title, pdf_name, abstract, paper_text, author_id, name AS author
    FROM papers
    JOIN paper_authors ON paper_authors.paper_id = papers.id
    JOIN authors ON authors.id = paper_authors.author_id
    ORDER BY papers.id
    ''', cnx)
    
    return df

def drop_duplicate_papers(df):
    '''
    Drops papers with the same id to remove duplicate papers
    Input: Pandas DataFrame
    Output: Pandas DataFrame
    '''
    return df.drop_duplicates(subset='id', keep='first') 

def clean(paper_text):
    '''
    Removes signs such as \n; Removes all non-character values; Make all text lowercase
    Input: string 
    Output: string
    '''
    import re
    text = re.sub('[^a-zA-Z]+', ' ', paper_text)
    for sign in ['\n', '\x0c']:
        text = text.replace(sign, ' ')
    return text.lower()

def clean_all_docs(df):
	'''
	Cleans all text in columns: paper_text, title, abstract
	Input: Pandas DataFrame
	Output: Pandas DataFrame
	'''
	df['paper_text_clean'] = df['paper_text'].apply(lambda x: clean(x))
	df['title_clean']      = df['title'].apply(lambda x: clean(x))
	df['abstract_clean']   = df['abstract'].apply(lambda x: clean(x))
	return df

def create_scaled_matrix(matrix):
    '''
    Scales matrix using StandardScaler
    Input: Numpy Array
    Output: Numpy Array
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)
