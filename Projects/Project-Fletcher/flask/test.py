import flask
import pandas as pd
import numpy as np


# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def hello():
    return "Flask App is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_pickle('../data/pipeline_data/nmf/tsne_df_2d_tfidf.pkl')
    data = flask.request.json

    year = int(data['year'])

    df_year = df[df['Year'] <= year]

    results = {'topic': list(df_year['Topic'].astype('float64')),
               'x': list(df_year['X']),
               'y': list(df_year['Y'])}

    return flask.jsonify(results)

@app.route('/test', methods=['POST'])
def test():
	df = pd.read_pickle('test.pkl')

	input_data = flask.request.json

	country = input_data['Country']

	ingredients = list(df[country])

	results = {'ingredients':ingredients}

	return flask.jsonify(results)

@app.route('/tiffany', methods=['POST'])
def tiffany():
  data = flask.request.json

  df = pd.read_pickle('dist_df.pkl')

  tag = str(data['tag'])
  d = dict(df[tag])
  sorted_dist = sorted(d.items(), key=lambda x: x[1])
  results = {'Recommendations': [sorted_dist[1], sorted_dist[2], sorted_dist[3]]}

  return flask.jsonify(results)


app.run(debug=True)

country = str(input_data['Country'])

ingredients = df[country]

results = {'Ingredients': ingredients}