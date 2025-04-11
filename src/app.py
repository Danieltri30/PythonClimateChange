from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = Flask(__name__)

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
        
        # Temperature over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='time', y='temperature')
        plt.title('Global Temperature Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature Deviations (°C)')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        temp_plot = get_plot_base64()
        plots['temperature'] = temp_plot
        
        # CO2 levels over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='time', y='co2_ppm')
        plt.title('CO2 Levels Over Time')
        plt.xlabel('Time')
        plt.ylabel('CO2 (ppm)')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        co2_plot = get_plot_base64()
        plots['co2'] = co2_plot
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        heatmap_plot = get_plot_base64()
        plots['heatmap'] = heatmap_plot
        
        return render_template('visualize.html', plots=plots)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True) 