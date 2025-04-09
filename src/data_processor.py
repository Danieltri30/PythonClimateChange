import pandas as pd
import numpy as np

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
        min_maxTemp = df["temperature_2m_max"].min()
        max_maxTemp = df["temperature_2m_max"].max()
        df["temperature_2m_max_normalized"] = (df["temperature_2m_max"] - min_maxTemp) / (max_maxTemp - min_maxTemp)

        min_minTemp = df["temperature_2m_min"].min()
        max_minTemp = df["temperature_2m_min"].max()
        df["temperature_2m_min_normalized"] = (df["temperature_2m_min"] - min_minTemp) / (max_minTemp - min_minTemp)

        return df

    # TODO: Implement this
    # def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Split data into features (year, month) and target (temperature)"""

if __name__ == "__main__":
    df = DataProcessor("../data/Temperature.csv").clean_data()
    print(df.head())

