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
import algorithms as algo
import visualizergui as viz
from threading import Thread , Lock
from flask import redirect, url_for
###

app = Flask(__name__, template_folder='templates')

def get_plot_base64():
    """Convert matplotlib plot to base64 string"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64

@app.route('/')
def index():
    return render_template('index.html')

training_results = {}
training_lock = Lock()
@app.route('/train-prediction')
def train_prediction():
    def background_train():
        try:
            atool = algo.GeneralTasks()
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
        # Read the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        df = pd.read_csv(csv_path)
        
        # Prepare data for clustering
        X = df[['temperature', 'co2_ppm']].values
        
        # Train clustering model
        clustering = Clustering()
        results = clustering.cluster_data(X)
        
        return render_template('training_results.html',
                             model_type='Clustering',
                             results=results)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

'''
@app.route('/gauge-model')
def gauge_model():
    try:
        # Read the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        df = pd.read_csv(csv_path)
        
        # Prepare data for evaluation
        X = df[['temperature', 'co2_ppm']].values
        y = df['temperature'].shift(-1).dropna().values
        X = X[:-1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Evaluate models
        prediction_model = Predictions(X_train, X_test, y_train, y_test)
        prediction_model.build_model()
        prediction_results = prediction_model.evaluate()
        
        clustering = Clustering()
        clustering_results = clustering.cluster_data(X)
        
        return render_template('model_evaluation.html',
                             prediction_results=prediction_results,
                             clustering_results=clustering_results)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/predict-future')
def predict_future():
    try:
        # Load the best model
        model = load_model('trained_rnn_model.h5')
        
        # Read the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        df = pd.read_csv(csv_path)
        
        # Prepare data for prediction
        X = df[['temperature', 'co2_ppm']].values[-1].reshape(1, -1)
        
        # Make predictions
        predictions = model.predict(X)
        
        return render_template('future_predictions.html',
                             predictions=predictions)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
'''

@app.route('/visualize')
def visualize():
    try:
        # Read the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        df = pd.read_csv(csv_path)
        
        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Create visualizations
        plots = {}
        
        # Temperature over time with Plotly
        fig_temp = px.line(df, x='time', y='temperature',
                          title='Global Temperature Over Time',
                          labels={'time': 'Time', 'temperature': 'Temperature Deviations (°C)'})
        fig_temp.update_layout(
            xaxis_title='Time',
            yaxis_title='Temperature Deviations (°C)',
            hovermode='x unified'
        )
        plots['temperature'] = fig_temp.to_html(full_html=False)
        
        # CO2 levels over time with Plotly
        fig_co2 = px.line(df, x='time', y='co2_ppm',
                         title='CO2 Levels Over Time',
                         labels={'time': 'Time', 'co2_ppm': 'CO2 (ppm)'})
        fig_co2.update_layout(
            xaxis_title='Time',
            yaxis_title='CO2 (ppm)',
            hovermode='x unified'
        )
        plots['co2'] = fig_co2.to_html(full_html=False)
        
        # Correlation heatmap with Plotly
        corr_matrix = df.corr()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig_heatmap.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        plots['heatmap'] = fig_heatmap.to_html(full_html=False)
        
        # Combined temperature and CO2 plot
        fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combined.add_trace(
            go.Scatter(x=df['time'], y=df['temperature'], name="Temperature"),
            secondary_y=False,
        )
        fig_combined.add_trace(
            go.Scatter(x=df['time'], y=df['co2_ppm'], name="CO2"),
            secondary_y=True,
        )
        fig_combined.update_layout(
            title_text="Temperature and CO2 Levels Over Time",
            xaxis_title="Time",
            hovermode='x unified'
        )
        fig_combined.update_yaxes(title_text="Temperature Deviations (°C)", secondary_y=False)
        fig_combined.update_yaxes(title_text="CO2 (ppm)", secondary_y=True)
        plots['combined'] = fig_combined.to_html(full_html=False)
        
        return render_template('visualize.html', plots=plots)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True) 