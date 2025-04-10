# pip install pandas
# pip install numpy
#pip install netCDF4

import pandas as pd
import numpy as np
from typing import Tuple
import xarray as xr

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    # This will merge our two datasets(.csv) , based on time and long/latitude
    def merge_data(self,frame1: pd.DataFrame ,frame2: pd.DataFrame):
        frame1["time"] = pd.to_datetime(frame1["time"]).dt.date
        frame2["time"] = pd.to_datetime(frame2["time"]).dt.date
        #LEts merge based on same time
        frame1_grouped = frame1.groupby("time")["temperature"].mean().reset_index()
        frame2_grouped = frame2.groupby("time")["co2_ppm"].mean().reset_index()
        frame1["time"] = frame1["time"].dt.to_period("M").dt.to_timestamp()
        frame2["time"] = frame2["time"].dt.to_period("M").dt.to_timestamp()
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
            if "temperature" in chunk.columns:
                chunk = chunk.dropna(subset=["temperature"])
            else:
                print("'temperature' column not found in this chunk")
            chunks.append(chunk)

        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            print(" No data loaded from:", self.file_path)
            return pd.DataFrame()  
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
   

def main():
    #util = DataProcessor("Land_and_Ocean_LatLong1.nc")
    #util.format_nc()
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
    fulldata.to_csv("Final-co2-berk.csv",index = False)



if __name__ == "__main__":
    main()


