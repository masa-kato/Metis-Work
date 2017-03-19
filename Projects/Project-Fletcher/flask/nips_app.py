import flask
import numpy as numpy
import pandas as pd


# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open('nips.html', 'r') as html_file:
        return html_file.read()


@app.route('/get_data', methods=['POST'])
def get_data():
	# Import DataFrame
	df = pd.read_pickle('../data/misc_models/nmf/tsne_df_perp100_n_iter5000.pkl')

	# Read data from POST request
	data = flask.request.json

	year = int(data['year'])

	# Perform tasks 
	df_year = df[df['Year'] <= year]

	results = {'x': list(df_year['X']),
               'y': list(df_year['Y']), 
               'topic': list(df_year['Topic'].astype('float64'))}

    # Return Results           
	return flask.jsonify(results)

app.run(host='0.0.0.0')
app.run(debug=True)