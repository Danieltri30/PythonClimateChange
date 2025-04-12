from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import data_processor as pros
import algorithmsgui as algo
import visualizergui as viz
from threading import Thread , Lock
from flask import redirect, url_for

app = Flask(__name__,template_folder='templates')
atool = algo.GeneralTasks()
vtool = viz.VisualizeData()

@app.route('/')
def index():
    return render_template('index.html')

atool = algo.GeneralTasks()
vtool = viz.VisualizeData()


def get_plot_base64():
    """Convert matplotlib plot to base64 string"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64

training_results = {}
training_lock = Lock()
@app.route('/train-prediction')
def train_prediction():
    def background_train():
        try:
            results = atool.run_prediction_model_for_gui()
            with training_lock:
                training_results['results'] = results
        except Exception as e:
            with training_lock:
                training_results['results'] = {'error': str(e)}

    Thread(target=background_train).start()

    # Redirect immediately while training happens in background
    return redirect(url_for('training_status'))

@app.route('/training-status')
def training_status():
    with training_lock:
        if 'results' in training_results:
            res = training_results.pop('results')  # clear for next run
            if 'error' in res:
                return f"An error occurred: {res['error']}", 500
            return render_template('trainin.html', **res)
    return render_template('trainin.html')


@app.route('/train-clustering')
def train_clustering():
    try:
        atool = algo.GeneralTasks()
        results = atool.run_cluster_simulation_gui()
        return render_template('clustering.html',
                               model_type='Clustering',
                               **results)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

best_training_results = {}
best_training_lock = Lock()
@app.route('/gauge-model')
def train_best_prediction():
    def besttrain():
        try:
            results = atool.run_best_prediction_model_for_gui()
            with best_training_lock:
                best_training_results['results'] = results
        except Exception as e:
            with best_training_lock:
                best_training_results['results'] = {'error': str(e)}
    Thread(target=besttrain).start()

    # Redirect immediately while training happens in background
    return redirect(url_for('best_training_status'))

@app.route('/beststatus')
def best_training_status():
    with best_training_lock:
        if 'results' in best_training_results:
            res = best_training_results.pop('results')  # clear for next run
            if 'error' in res:
                return f"An error occurred: {res['error']}", 500
            return render_template('best.html', **res)
    return render_template('best.html')


@app.route('/predict-future')
def predict_future():
    try:
        # Get the future data and predictions
        future_data_holder, predictions = atool.create_future_data()

        # Generate the temperature plot in base64 format
        plot_base64 = vtool.predicted_temperature_levels_over_time_for_gui(future_data_holder)

        # Pass the plot to the template
        return render_template('predict_future.html', plot_base64=plot_base64)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500


@app.route('/visualize')
def visualize():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        final_df = pd.read_csv(csv_path)
        visualizations = vtool.general_data_analysis_for_gui(final_df)
        return render_template('visualize.html',**visualizations)  # Replace 'visualize.html' with your actual template name
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

#@app.route('/about')
#def about():
    #return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 