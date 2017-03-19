import luigi
import pickle
import pandas as pd
from preprocess import *
from vectorize import *
from nmf import *
from tsne import *
from lda import *

class ImportData(luigi.Task):
	def requires(self):
		return [] 

	def output(self):
		return luigi.LocalTarget('../data/pipeline_data/df.pkl')

	def run(self):
		df = import_raw_data()
		df.to_pickle('../data/pipeline_data/df.pkl')

class CleanData(luigi.Task):
	def requires(self):
		return ImportData()

	def output(self):
		return luigi.LocalTarget('../data/pipeline_data/clean_df.pkl')

	def run(self):
		df = pd.read_pickle('../data/pipeline_data/df.pkl')
		df = drop_duplicate_papers(df)
		df = clean_all_docs(df)

		df.to_pickle('../data/pipeline_data/clean_df.pkl')

class Vectorize(luigi.Task):
	def requires(self):
		return CleanData()

	def output(self):
		return [luigi.LocalTarget('../data/pipeline_data/count_vectorizer.p'),
		        luigi.LocalTarget('../data/pipeline_data/count_sparse.p'),
		        luigi.LocalTarget('../data/pipeline_data/tfidf_vectorizer.p'), 
		        luigi.LocalTarget('../data/pipeline_data/tfidf_sparse.p')]

	def run(self):
		df = pd.read_pickle('../data/pipeline_data/clean_df.pkl')

		###################################################
		# Define Parameters (Alter values as necessary)
		ngram, num_features = 1, 1000
		###################################################

		# Create Vectorizer & Sparse Matrix
		count_vectorizer, count_sparse = create_count(df=df, column_name='paper_text_clean', num_features=num_features, ngram=ngram)
		tfidf_vectorizer, tfidf_sparse = create_tfidf(df=df, column_name='paper_text_clean', num_features=num_features, ngram=ngram)

		# Pickle Files
		names   = ['count_vectorizer','count_sparse','tfidf_vectorizer','tfidf_sparse']
		outputs = [ count_vectorizer,  count_sparse,  tfidf_vectorizer,  tfidf_sparse]
		for i in range(len(names)):
			file_path = '../data/pipeline_data/{}.p'.format(names[i])
			with open(file_path,'wb') as _out:
			    pickle.dump(outputs[i], _out)

class Nmf(luigi.Task):
	def requires(self):
		return Vectorize() 

	def output(self):
		return [luigi.LocalTarget('../data/pipeline_data/nmf/nmf_model_tfidf.p'),
				luigi.LocalTarget('../data/pipeline_data/nmf/nmf_topicspace_tfidf.p'),
				luigi.LocalTarget('../data/pipeline_data/nmf/nmf_topicspace_tfidf_scaled.p')]

	def run(self):
		with open('../data/pipeline_data/tfidf_vectorizer.p','rb') as _in:
			tfidf_vectorizer = pickle.load(_in)
		with open('../data/pipeline_data/tfidf_sparse.p','rb') as _in:
			tfidf_sparse = pickle.load(_in)

		###################################################
		# Define Parameters (Alter values as necessary)
		num_topics, lambda_reg, l1_ratio = 10, 0.1, 0.5
		###################################################

		# Create Model
		model, nmf_topicspace_tfidf  = create_nmf_model(sparse_matrix=tfidf_sparse, num_topics=num_topics, lambda_reg=lambda_reg, l1_ratio=l1_ratio)
		nmf_topicspace_tfidf_scaled = create_scaled_matrix(nmf_topicspace_tfidf)

		# Pickle Model
		with open('../data/pipeline_data/nmf/nmf_model_tfidf.p','wb') as _out:
			pickle.dump(model, _out)

		with open('../data/pipeline_data/nmf/nmf_topicspace_tfidf.p','wb') as _out:
			pickle.dump(nmf_topicspace_tfidf, _out)

		with open('../data/pipeline_data/nmf/nmf_topicspace_tfidf_scaled.p','wb') as _out:
			pickle.dump(nmf_topicspace_tfidf_scaled, _out)

class Tsne_for_Nmf(luigi.Task):
	def requires(self):
		return Nmf()

	def output(self):
		return [luigi.LocalTarget('../data/pipeline_data/nmf/tsne_model_2d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/nmf/tsne_matrix_2d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/nmf/tsne_model_3d_tfidf.p'), 
		        luigi.LocalTarget('../data/pipeline_data/nmf/tsne_matrix_3d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/nmf/tsne_df_2d_tfidf.pkl'),
		        luigi.LocalTarget('../data/pipeline_data/nmf/tsne_df_3d_tfidf.pkl')]

	def run(self):
		df = pd.read_pickle('../data/pipeline_data/clean_df.pkl')
		df.reset_index(drop=True, inplace=True)
		
		with open('../data/pipeline_data/nmf/nmf_topicspace_tfidf_scaled.p','rb') as _in:
			nmf_topicspace_tfidf = pickle.load(_in)

		###################################################
		# Define Parameters (Alter values as necessary)
		perplexity, random_state, metric = 30, 0, 'euclidean'
		###################################################

		# Create Model
		tsne_model_2d_tfidf, tsne_matrix_2d_tfidf = create_tsne_model(nmf_topicspace_tfidf, dimensions=2, random_state=random_state, metric=metric, perplexity=perplexity)
		tsne_model_3d_tfidf, tsne_matrix_3d_tfidf = create_tsne_model(nmf_topicspace_tfidf, dimensions=3, random_state=random_state, metric=metric, perplexity=perplexity)

		tsne_df_2d_tfidf = create_tsne_df_2D(tsne_matrix_2d_tfidf, nmf_topicspace_tfidf, df)
		tsne_df_3d_tfidf = create_tsne_df_3D(tsne_matrix_3d_tfidf, nmf_topicspace_tfidf, df)

		# Pickle Files
		names   = ['tsne_model_2d_tfidf', 'tsne_matrix_2d_tfidf', 'tsne_model_3d_tfidf', 'tsne_matrix_3d_tfidf']
		outputs = [ tsne_model_2d_tfidf ,  tsne_matrix_2d_tfidf ,  tsne_model_3d_tfidf ,  tsne_matrix_3d_tfidf ]
		for i in range(len(names)):
			file_path = '../data/pipeline_data/nmf/{}.p'.format(names[i])
			with open(file_path,'wb') as _out:
			    pickle.dump(outputs[i], _out)

		tsne_df_2d_tfidf.to_pickle('../data/pipeline_data/nmf/tsne_df_2d_tfidf.pkl')
		tsne_df_3d_tfidf.to_pickle('../data/pipeline_data/nmf/tsne_df_3d_tfidf.pkl')

class Lda(luigi.Task):
	def requires(self):
		return Vectorize() 

	def output(self):
		return [luigi.LocalTarget('../data/pipeline_data/lda/lda_model_tfidf.p'),
				luigi.LocalTarget('../data/pipeline_data/lda/lda_topicspace_tfidf.p'),
				luigi.LocalTarget('../data/pipeline_data/lda/lda_topicspace_tfidf_scaled.p')]

	def run(self):
		with open('../data/pipeline_data/tfidf_vectorizer.p','rb') as _in:
			tfidf_vectorizer = pickle.load(_in)
		with open('../data/pipeline_data/tfidf_sparse.p','rb') as _in:
			tfidf_sparse = pickle.load(_in)

		###################################################
		# Define Parameters (Alter values as necessary)
		num_topics = 10
		###################################################

		# Create Model
		model, lda_topicspace_tfidf  = create_lda_model(sparse_matrix=tfidf_sparse, num_topics=num_topics)
		lda_topicspace_tfidf_scaled = create_scaled_matrix(lda_topicspace_tfidf)

		# Pickle Model
		with open('../data/pipeline_data/lda/lda_model_tfidf.p','wb') as _out:
			pickle.dump(model, _out)

		with open('../data/pipeline_data/lda/lda_topicspace_tfidf.p','wb') as _out:
			pickle.dump(lda_topicspace_tfidf, _out)

		with open('../data/pipeline_data/lda/lda_topicspace_tfidf_scaled.p','wb') as _out:
			pickle.dump(lda_topicspace_tfidf_scaled, _out)

class Tsne_for_Lda(luigi.Task):
	def requires(self):
		return Lda()

	def output(self):
		return [luigi.LocalTarget('../data/pipeline_data/lda/tsne_model_2d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/lda/tsne_matrix_2d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/lda/tsne_model_3d_tfidf.p'), 
		        luigi.LocalTarget('../data/pipeline_data/lda/tsne_matrix_3d_tfidf.p'),
		        luigi.LocalTarget('../data/pipeline_data/lda/tsne_df_2d_tfidf.pkl'),
		        luigi.LocalTarget('../data/pipeline_data/lda/tsne_df_3d_tfidf.pkl')]

	def run(self):
		df = pd.read_pickle('../data/pipeline_data/clean_df.pkl')
		df.reset_index(drop=True, inplace=True)
		
		with open('../data/pipeline_data/lda/lda_topicspace_tfidf.p','rb') as _in:
			lda_topicspace_tfidf = pickle.load(_in)

		###################################################
		# Define Parameters (Alter values as necessary)
		perplexity, random_state, metric = 30, 0, 'euclidean'
		###################################################

		# Create Model
		tsne_model_2d_tfidf, tsne_matrix_2d_tfidf = create_tsne_model(lda_topicspace_tfidf, dimensions=2, random_state=random_state, metric=metric, perplexity=perplexity)
		tsne_model_3d_tfidf, tsne_matrix_3d_tfidf = create_tsne_model(lda_topicspace_tfidf, dimensions=3, random_state=random_state, metric=metric, perplexity=perplexity)

		tsne_df_2d_tfidf = create_tsne_df_2D(tsne_matrix_2d_tfidf, lda_topicspace_tfidf, df)
		tsne_df_3d_tfidf = create_tsne_df_3D(tsne_matrix_3d_tfidf, lda_topicspace_tfidf, df)

		# Pickle Files
		names   = ['tsne_model_2d_tfidf', 'tsne_matrix_2d_tfidf', 'tsne_model_3d_tfidf', 'tsne_matrix_3d_tfidf']
		outputs = [ tsne_model_2d_tfidf ,  tsne_matrix_2d_tfidf ,  tsne_model_3d_tfidf ,  tsne_matrix_3d_tfidf ]
		for i in range(len(names)):
			file_path = '../data/pipeline_data/lda/{}.p'.format(names[i])
			with open(file_path,'wb') as _out:
			    pickle.dump(outputs[i], _out)

		tsne_df_2d_tfidf.to_pickle('../data/pipeline_data/lda/tsne_df_2d_tfidf.pkl')
		tsne_df_3d_tfidf.to_pickle('../data/pipeline_data/lda/tsne_df_3d_tfidf.pkl')


if __name__ == '__main__':
	luigi.run(["--local-scheduler"], main_task_cls=Tsne_for_Lda)
