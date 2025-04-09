from os import stat
import matplotlib.pyplot as plt
from typing import List, Tuple 

class Visualizer: 
    @staticmethod
    def plot_temperature_trend(years: List[int], temperatures: List[float], predictions: List[float]) -> None:
        plt.figure(figsize=(5, 5))
        plt.plot(years, temperatures, label="Actual")
        plt.plot(years, predictions, label="Predicted")
        plt.xlabel("Year")
        plt.ylabel("Temperature (normalized")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_clustered_data(data: List[Tuple[float, float]], labels: List[int]) -> None:
        pass

    @staticmethod
    def plot_anomalies(time_series: List[float], anomalies: List[bool]) -> None:
        pass
