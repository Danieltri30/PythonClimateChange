import pandas as pd
import numpy as np
from typing import Tuple

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        pass

    def load_data(self) -> pd.DataFrame:
        """Load climate data from CSV file"""
        return pd.read_csv(self.file_path)

    def clean_data(self) -> pd.DataFrame:
        """Remove rows with missing values and normalize temperature data"""
        df = self.load_data().dropna()
        if "ByCountry" in self.file_path:
            df["AverageTemperatureNormalized"] = (df["AverageTemperature"] - df.groupby("Country")["AverageTemperature"].transform("min")) / (df.groupby("Country")["AverageTemperature"].transform("max") - df.groupby("Country")["AverageTemperature"].transform("min"))
            # df["MinAvg"] = df.groupby("Country")["AverageTemperature"].transform("min")
            # df["MaxAvg"] = df.groupby("Country")["AverageTemperature"].transform("max")
        elif "ByMajorCity" in self.file_path:
            df["AverageTemperatureNormalized"] = (df["AverageTemperature"] - df.groupby(["Country", "City"])["AverageTemperature"].transform("min")) / (df.groupby(["Country", "City"])["AverageTemperature"].transform("max") - df.groupby(["Country", "City"])["AverageTemperature"].transform("min"))
            # df["MinAvg"] = df.groupby(["Country", "City"])["AverageTemperature"].transform("min")
            # df["MaxAvg"] = df.groupby(["Country", "City"])["AverageTemperature"].transform("max")
        return df

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into features (country, [city,] month, day) and target (average temperature)"""
        df = self.clean_data()
        if "ByCountry" in self.file_path or "ByMajorCity" in self.file_path:
            df["year"] = pd.to_numeric(df["dt"].astype(str).str.split("-").str[0])
            df["month"] = pd.to_numeric(df["dt"].astype(str).str.split("-").str[1])
            df["day"] = pd.to_numeric(df["dt"].astype(str).str.split("-").str[2])
            target = df["AverageTemperature"].to_numpy()

        if "ByCountry" in self.file_path:
            features = df[["Country", "year", "month", "day"]].to_numpy()
        elif "ByMajorCity" in self.file_path:
            features = df[["City", "Country", "year", "month", "day"]].to_numpy()
        elif "Deviation" in self.file_path:
            features = df["Year"].to_numpy()
            target = df[["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "J-D"]].to_numpy()

        return features, target



if __name__ == "__main__":
    # country_df = DataProcessor("../data/GlobalLandTemperaturesByCountry.csv").clean_data()
    # print(country_df.head())
    # print(country_df.tail())

    # city_df = DataProcessor("../data/GlobalLandTemperaturesByMajorCity.csv").clean_data()
    # print(city_df.head())
    # print(city_df.tail())

    # global_df = DataProcessor("../data/GlobalTemperatureDeviation.csv").clean_data()
    # print(global_df.head())
    # print(global_df.tail())

    # features, target = DataProcessor("../data/GlobalLandTemperaturesByMajorCity.csv").get_features_and_target()
    # count = 0
    # for i, j in zip(features, target):
    #     count += 1
    #     if count == 10:
    #         break
    #     print(f"{i}\t{j}")

    # features, target = DataProcessor("../data/GlobalLandTemperaturesByCountry.csv").get_features_and_target()
    # count = 0
    # for i, j in zip(features, target):
    #     count += 1
    #     if count == 10:
    #         break
    #     print(f"{i}\t{j}")

    # features, target = DataProcessor("../data/GlobalTemperatureDeviation.csv").get_features_and_target()
    # count = 0
    # for i, j in zip(features, target):
    #     count += 1
    #     if count == 10:
    #         break
    #     print(f"{i}\t{j}")
    pass


