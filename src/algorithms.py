#pip install pandas
#pip install matplotlib
#pip install numpy
#pip install seaborn
#pip install scikit-learn
#pip install cartopy
#pip install keras
#pip install keras-tuner
#pip install tensorflow

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import random
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Modeling tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import keras_tuner as kt
import tensorflow as tf



class GeneralModeling:
    def inverse_transform(self, temp_scaled_val: float, co2_scaled_val: float) -> tuple:
        # Reshape to (1,1) , lets hope the model takes this shape
        temp_orig = self.temp_scaler.inverse_transform([[temp_scaled_val]])[0][0]
        co2_orig = self.co2_scaler.inverse_transform([[co2_scaled_val]])[0][0]
        return temp_orig, co2_orig

class Predictions(GeneralModeling):
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None  # Initialize as None

    def build_model(self):
        self.model = build_model(None)  # Use the standalone build_model function

    def train(self, epochs=45, batch_size=32):
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
        )
        self.model.save("trained_rnn_model.h5")
        print("Model saved as 'trained_rnn_model.h5'")

    def evaluate(self):
        result = self.model.evaluate(self.X_test, self.y_test)
        if len(result) == 3:  
            loss, mae, r2 = result
            print(f"Loss: {loss}, MAE: {mae}, R²: {r2}")
        else:
            loss = result
            print(f"Loss: {loss}")

class Clustering(GeneralModeling):
    def __init__(self):
        super().__init__()
        # Add any initialization code here

    def cluster_data(self):
        # Add clustering implementation here
        pass

# Main function
def main():
    # Perform hyperparameter tuning with Hyperband
    tuner = kt.Hyperband(
        build_model,  # Reference the standalone function
        objective='val_mae',
        max_epochs=45,
        directory='tuner_results',
        project_name='apartment_model',
    )

    tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps.values}")

    # Use the NeuralNetwork class for training and evaluation
    nn = NeuralNetwork(X_train, X_test, y_train, y_test)
    nn.build_model()
    nn.train()
    nn.evaluate()

if __name__ == "__main__":
    main()