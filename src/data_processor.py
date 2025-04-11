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

# FOr some reason the Berkley data was in fractional year format, so we use this to convert it to a format
# that aligns with the CO2 dataset
def convertyear(yf):
        year = int(yf)
        rem = yf - year
        base = datetime.datetime(year, 1, 1)
        result = base + datetime.timedelta(days=rem * 365.25)
        return result.date()

# SInce we dont want to work with DAILY Co2 changes, we will average CO2 emissions over the month with this function
def average_monthly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["time"] = df["time"].dt.to_period("M").dt.to_timestamp()
        grouped = df.groupby("time")[value_col].mean().reset_index()
        return grouped 

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
        
        # Set the x-ticks on the plot with rotation
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
        
        # Set the x-ticks on the plot with rotation
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
class FineTuneData:
    
    #Takes in the dataframe we will be fine tuning (Normalization and processing for the algorithm to better understand)
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        #Normalize with min max scaler , not using one hot encoding since no categories
        self.temp_scaler = MinMaxScaler()
        self.co2_scaler = MinMaxScaler()
        self.scaled_df = None
    def scaledata(self) -> pd.DataFrame:
        # Transform each piece of data independently
        self.df['temperature_scaled'] = self.temp_scaler.fit_transform(self.df[['temperature']])
        self.df['co2_scaled'] = self.co2_scaler.fit_transform(self.df[['co2_ppm']])
        
        # Store the result
        self.scaled_df = self.df
        return self.scaled_df
    
    def get_scalers(self):
        return self.temp_scaler, self.co2_scaler

    def inverse_transform(self, temp_scaled_val: float, co2_scaled_val: float) -> tuple:
        # Reshape to (1,1) , lets hope the model takes this shape
        temp_orig = self.temp_scaler.inverse_transform([[temp_scaled_val]])[0][0]
        co2_orig = self.co2_scaler.inverse_transform([[co2_scaled_val]])[0][0]
        return temp_orig, co2_orig



class DataProcessor:
    #Intialize DataProcessor instance including file path to dataset
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    # This will merge our two datasets(.csv) , based on time and long/latitude
    def merge_data(self,frame1: pd.DataFrame ,frame2: pd.DataFrame):
        frame1["time"] = pd.to_datetime(frame1["time"]).dt.to_period("M").dt.to_timestamp()
        frame2["time"] = pd.to_datetime(frame2["time"]).dt.to_period("M").dt.to_timestamp()

    # Group by time and average
        frame1_grouped = frame1.groupby("time")["temperature"].mean().reset_index()
        frame2_grouped = frame2.groupby("time")["co2_ppm"].mean().reset_index()
        print("Frame1 unique times:", frame1["time"].sort_values().unique()[:10])
        print("Frame2 unique times:", frame2["time"].sort_values().unique()[:10])
        print("Frame1 date range:", frame1["time"].min(), "to", frame1["time"].max())
        print("Frame2 date range:", frame2["time"].min(), "to", frame2["time"].max())
            # Merged based on only time since we want global climate change for predictor atleast
        merged_data = pd.merge(frame1_grouped,frame2_grouped , on="time",how="inner")
        if merged_data.empty:
            print("Merging was a failure, you are cooked.")
        else:
            print("Merging was sucessful!")
        return merged_data
    
    # Currently co2 data has place holders for the lat and lon, this fixes that
    def get_true_location(self) -> pd.DataFrame:
        df = self.load_data()
        if "LatDim" in df.columns and "LonDim" in df.columns:
            from netCDF4 import Dataset
            tempd = Dataset(self.file_path)
            lats = tempd.variables['latitude'][:]
            lons = tempd.variables['longitude'][:]

            df["latitude"] = df["LatDim"].map(lambda i: lats[i])
            df["longitude"] = df["LonDim"].map(lambda i: lons[i]) 

            df.drop(["LatDim","LonDim"],axis=1,inplace=True)
        else:
            print("You are cooked buddy, we cant find these columns")
        return df        


    '''def dropNAdata(self) -> pd.DataFrame:
        df = self.load_data()
        return df.dropna(subset=["temperature"])'''

    def load_data(self) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(self.file_path, chunksize=500000):
            print(" Columns in this chunk:", chunk.columns.tolist())
            print(" Chunk loaded:", len(chunk))

            if "time" in chunk.columns:
                chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")

            if "temperature" in chunk.columns:
                chunk = chunk.dropna(subset=["temperature"])

            chunks.append(chunk)

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            print("Final time range:", df["time"].min(), "to", df["time"].max())
            print("Unique time points:", df["time"].nunique())
            return df
        else:
            print(" No data loaded from:", self.file_path)
            return pd.DataFrame()
    
    #I dont even wanna talk about this , .nc is a horrible format
    # Esentially opening the .nc which is like a zipped file , extracting about 100,000+ lines and then conerting it to a .csv
    # Note that we are parsing chunks of data at a time since the data is massive and extremly resourcei ntensive
    def format_nc(self):
        ds = xr.open_dataset("Land_and_Ocean_LatLong1.nc")

        times = ds["time"].values
        chunks = []

        for i, t in enumerate(times):
            print(f"Processing {i+1}/{len(times)}: {t}")
            sub = ds.sel(time=t)["temperature"]
            df = sub.to_dataframe().reset_index()
            df = df.dropna(subset=["temperature"])
            chunks.append(df)

        full_df = pd.concat(chunks, ignore_index=True)
        print("Date range:", full_df["time"].min(), "to", full_df["time"].max())
        print("Unique time points:", full_df["time"].nunique())
        full_df.to_csv("Berkley_temperature_full.csv", index=False)
    # Since daily tracking will be extremely hard, WE will average it to a month.   
     
    

def main():
    holder = 100
    if holder == 0:
        #util = DataProcessor("Land_and_Ocean_LatLong1.nc")
        #util.format_nc()
        #df = pd.read_csv("Berkley_temperature_full.csv", parse_dates=["time"])
        #print("Time range:", df["time"].min(), "to", df["time"].max())
        util = DataProcessor("random")
        df_co2 = pd.read_csv("data/co2_processed.csv")
        df_co2["time"] = pd.to_datetime(df_co2["time"], errors="coerce")
        print("Time range for Co2 data:", df_co2["time"].min().date(), "to", df_co2["time"].max().date())
        df_berk = pd.read_csv("data/Berkley_temperature_full.csv")
        print("Time range:", df_berk["time"].min(), "to", df_berk["time"].max())
        df_berk["time"] = df_berk["time"].apply(convertyear)
        print("Time range after convert year is applied:", df_berk["time"].min(), "to", df_berk["time"].max())
        df_berk["time"] = pd.to_datetime(df_berk["time"], errors="coerce")
        print("Time range for berkley data:", df_berk["time"].min().date(), "to", df_berk["time"].max().date())
        #Now our time is properly formatted in date time
        
        print(df_co2.columns)
        print(df_berk.columns)

        #SInce we want global average for the predictor(not cluster), We will drop lon and lat
        df_berk = df_berk.drop(columns=["latitude", "longitude"], errors="ignore")
        df_co2 = df_co2.drop(columns=["latitude", "longitude"], errors="ignore")

        #Lets check to see if it occured
        print("Status of your datasets after drops")
        print(df_co2.columns)
        print(df_berk.columns)

        avg_temp = average_monthly(df_berk, "temperature")
        avg_co2 = average_monthly(df_co2, "co2_ppm")

        print(avg_temp.head())
        print(avg_co2.head())
            # Commented it out to perserve the structure of our final dataset, but still shown to show process
        #final_df = pd.merge(avg_temp, avg_co2, on="time", how="inner")
        #print(final_df.head())
        #final_df.to_csv("data/FinalProcessedData.csv",index = False)
    elif holder == 1:
        final_df = pd.read_csv("data/FinalProcessedData.csv")
        #Lets make sure our data is not damaged in the process of setting it to dataframe
        print(final_df.head(10))
        print("End of head count 10")
        processor = FineTuneData(final_df)
        scaled_df = processor.scaledata()
        print(scaled_df.head())

        # Lets test the de-normalization function which we will use 
        # TO read data later
        temp_raw, co2_raw = processor.inverse_transform(0.7, 0.8)
        print("Recovered values:", temp_raw, co2_raw)
        #Lets save these scalers to continue this over at algorithms.py
        joblib.dump(processor.get_scalers()[0], "temp_scaler.pkl")
        joblib.dump(processor.get_scalers()[1], "co2_scaler.pkl")
        scaled_df.to_csv("Scaled_Data_For_Model.csv", index=False)
    elif holder == 2:

        final_df = pd.read_csv("data/FinalProcessedData.csv")
        dupes = final_df[final_df.duplicated()]
        #FInal check for general duplicates
        print("Exact Dulicate rows:\n",dupes)  
        #Final check for time duplicate
        time_dupes = final_df[final_df.duplicated(subset=["time"])]
        print("Duplicate timestamps:\n", time_dupes)
    else:
        final_df = pd.read_csv("data/FinalProcessedData.csv")
        plotplacer = VisualizeData()
        plotplacer.co2_vs_Temperature(final_df)
        #From the graph we see a clear correlation between
        # the levels of CO2 and temperature deviation
        # It Seems that as temperature deviates in a warmer sense of things
        # CO2 emissions also equally rise, while a deviation towards global cooling
        # displays CO2 emissions being near the average level
        plotplacer.co2_over_time(final_df)
        plotplacer.temperature_levels_over_time(final_df)


    

    '''
    berkleypath= "data/Berkley_temperature_full.csv"
    co2csvpath = "data/co2_processed.csv"
    # Create two dataprocessor utility instances
    utilco2 = DataProcessor(co2csvpath)
    utilberkley = DataProcessor(berkleypath)
    # Load data into seperate dataframes using util
    df_co2 = utilco2.load_data()
    df_berk = utilberkley.load_data()
    # comine both on time
    fulldata = utilberkley.merge_data(df_berk,df_co2)
    print("Merging has passed")
    print(fulldata.head())
    # Send near finalized data to a .csv
    fulldata.to_csv("Final-co2-berk.csv",index = False)'''



if __name__ == "__main__":
    main()
