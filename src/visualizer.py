# pip install pandas
# pip install numpy
#pip install netCDF4
#pip install sklearn
#pip install joblib

import matplotlib
import joblib
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas as pd
import numpy as np
from typing import Tuple
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, KFold
import os

class VisualizeData:
    #Shows how CO2_ppm changes over itme
    #Co2_ppm(Parts per Million) means how many molecules of carbon dioxide there are in every million molecules of air.
    def co2_over_time(self,df: pd.DataFrame):
        plt.figure(figsize=[10, 6])

        subdf = df
        new = set()
        for val in subdf["time"]:
            temp = val[:4]
            new.add(temp)
        for val in new:
            print(val)

        sset = sorted(new)        
        # Plot CO2 levels over time (no changes to the CO2 data)
        plt.plot(df['time'], df['co2_ppm'], color='red', label='CO2 Levels (ppm)')
        
        min_year = 0
        max_year = 2000
        
        xticks = range(min_year, max_year + 1, 1000)
        
        plt.xticks(xticks, rotation=45)
        
        plt.title('CO2 PPM Level Over Time')
        plt.xlabel('Time (Every 50 Years)')
        plt.ylabel('CO2 Levels (ppm)')
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # To avoid clipping of labels
        
        plt.show()

    #Shows Temperature Deviation over time
    #Temperature deviation refers to sudden alterations of temperature vs the average for a chosen period of time
    def temperature_levels_over_time(self,df:pd.DataFrame):
        plt.figure(figsize=[10, 6])

        subdf = df
        new = set()
        for val in subdf["time"]:
            temp = val[:4]
            new.add(temp)
        for val in new:
            print(val)

        sset = sorted(new)        
        # Plot CO2 levels over time (no changes to the CO2 data)
        plt.plot(df['time'], df['temperature'], color='red', label='CO2 Levels (ppm)')
        
        min_year = 0
        max_year = 2000
        
        xticks = range(min_year, max_year + 1, 1000)
        
        plt.xticks(xticks, rotation=45)
        
        plt.title('Temperature Deviations over time')
        plt.xlabel('Time (Every 50 Years)')
        plt.ylabel('CO2 Levels (ppm)')
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  
        
        plt.show()                       

    #Scatter plot to show correlation between temperature and CO2 Levels
    def co2_vs_Temperature(self,df:pd.DataFrame):
        plt.figure(4)
        plt.scatter(df['temperature'], df['co2_ppm'], alpha=0.5)
        plt.title('CO2 Level vs Temperature')
        plt.xlabel('Temperature Deviation')
        plt.ylabel('CO2 Level (ppm)')
        plt.show()

    #Generalized function to show cluster visuals
    #Robust and accepts various clustering algorithms
    def cluster_visualization(self,df,s):
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='viridis')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"City Clusters based on Temp + Coordinates ({s})")
        plt.colorbar(label='Cluster')
        plt.show()

    #Proves that our choice of k was perfect mathematically
    def showelbow(self,klist):         
        plt.plot(range(1, 11), klist, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    #Functions to keep code clean
    def general_data_analysis(self,df):
        print("START OF GENERAL DATA ANALYSIS:\n")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        final_df = pd.read_csv(csv_path)
        plotplacer = VisualizeData()
        plotplacer.co2_vs_Temperature(final_df)
        #From the graph we see a clear correlation between
        # the levels of CO2 and temperature deviation


        # It Seems that as temperature deviates in a warmer sense of things
        # CO2 emissions also equally rise, while a deviation towards global cooling
        # displays CO2 emissions being near the average level

        plotplacer.co2_over_time(final_df)

        #THis will show temperature levels over time
        plotplacer.temperature_levels_over_time(final_df)
        print("END OF GENERAL DATA ANALYSIS:\n")

    def oldClustercode():
        #Unused since we added a better version of this in algorithms
        print("START OF CLUSTERED DATA ANALYSIS:\n")
        #NOW LETS Look at plots for the CLUSTERING PREDICTIONS
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        #csv_path = os.path.join(script_dir, "..", "data", "FinalizedTrainedClusterData.csv")
        #city_df = pd.read_csv(csv_path)

        #Run function to vizualze Data
        #plotplacer.cluster_visualization(city_df)
        print("END OF CLUSTERED DATA ANALYSIS:\n")

    def evaluate_prediction_model(self,history):
        #Shows MAE through epochs
        plt.figure()
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('MAE over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # THis shows R^2 through epochs
        plt.figure()
        plt.plot(history.history['r2_metric'], label='Train R²')
        plt.plot(history.history['val_r2_metric'], label='Validation R²')
        plt.xlabel('Epochs')
        plt.ylabel('R² Score')
        plt.title('R² over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
    final_df = pd.read_csv(csv_path)
    plotplacer = VisualizeData()
    plotplacer.general_data_analysis(final_df)


if __name__ == '__main__':
    main()    