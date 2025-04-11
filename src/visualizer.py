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
        
        plt.title('CO2 Levels Over Time')
        plt.xlabel('Time (Every 50 Years)')
        plt.ylabel('CO2 Levels (ppm)')
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # To avoid clipping of labels
        
        plt.show()

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

    def co2_vs_Temperature(self,df:pd.DataFrame):
        plt.figure(4)
        plt.scatter(df['temperature'], df['co2_ppm'], alpha=0.5)
        plt.title('CO2 Level vs Temperature')
        plt.xlabel('Temperature Deviation')
        plt.ylabel('CO2 Level (ppm)')
        plt.show() 

def main():
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

if __name__ == '__main__':
    main()    