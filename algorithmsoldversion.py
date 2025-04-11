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
import pandas as pd
# Modeling tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import keras_tuner as kt
import tensorflow as tf

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + K.epsilon()))
    return r2

def make_sequences(data, seq_len=12):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[['time_scaled', 'co2_scaled']].iloc[i:i+seq_len].values)
            y.append(data['temperature_scaled'].iloc[i+seq_len])
        return np.array(X), np.array(y)

'''
def build_model(hp):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(12,2)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae', r2_metric]
        )
        model.summary()
        return model
'''
def build_model(hp):
    model = Sequential()

    # Tune LSTM manually
    model.add(LSTM(
        units=hp.Int('units_lstm_1', min_value=64, max_value=256, step=32),
        return_sequences=True,
        input_shape=(12, 2)
    ))

    # Tune number of units in the second LSTM layer
    model.add(LSTM(
        units=hp.Int('units_lstm_2', min_value=32, max_value=128, step=16),
        return_sequences=False
    ))

    # Tune size of dense layer
    model.add(Dense(hp.Int('dense_units', min_value=10, max_value=50, step=5)))

    # Output layer
    model.add(Dense(1))

    # Tune learning rate
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae', r2_metric]
    )

    return model

class GeneralModeling:
    def inverse_transform(self, temp_scaled_val: float, co2_scaled_val: float) -> tuple:
        # Reshape to (1,1) , lets hope the model takes this shape
        temp_orig = self.temp_scaler.inverse_transform([[temp_scaled_val]])[0][0]
        co2_orig = self.co2_scaler.inverse_transform([[co2_scaled_val]])[0][0]
        return temp_orig, co2_orig
    
    def normalizedata(self,final_df:pd.DataFrame, output_path: str="data/Scaled_Data_For_Model.csv") -> pd.DataFrame:
        final_df['time'] = pd.to_datetime(final_df['time'])
        start_date = final_df['time'].min()
        final_df['time_ordinal'] = (final_df['time'] - start_date).dt.days

        time_scaler = MinMaxScaler()
        temp_scaler = MinMaxScaler()
        co2_scaler = MinMaxScaler()

        # Apply scaling
        final_df['time_scaled'] = time_scaler.fit_transform(final_df[['time_ordinal']])
        final_df['temperature_scaled'] = temp_scaler.fit_transform(final_df[['temperature']])
        final_df['co2_scaled'] = co2_scaler.fit_transform(final_df[['co2_ppm']])
        final_df.to_csv(output_path, index=False)
        print(f"Saved Normalized data to :  {output_path}")

    def Split_data(self,df: pd.DataFrame):
        x = df[['time_scaled','co2_scaled']]
        y = df[['temperature_scaled']]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
    def grab_datasets(self):
        lis = []
        a = pd.read_csv("data/X_train.csv")
        b = pd.read_csv("data/y_train.csv")
        c = pd.read_csv("data/X_test.csv") 
        d = pd.read_csv("data/y_test.csv") 
        return a,b,c,d          

class Predictions(GeneralModeling):
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None  # Initialize as None
    def build_model(self):
        self.model = build_model(None)
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
    util = GeneralModeling()
    holder = 100
    if holder == 0:
        final_df = pd.read_csv("data/FinalProcessedData.csv")
        util.normalizedata(final_df)
    elif holder == 1:   
        finalscaled_df = pd.read_csv("data/Scaled_Data_For_Model.csv")
        util.Split_data(finalscaled_df)
    else:
        df = pd.read_csv("data/Scaled_Data_For_Model.csv")
        X,y = make_sequences(df,seq_len=12)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
        tuner = kt.Hyperband(
            build_model,  
            objective='val_mae',
            max_epochs=45,
            directory='tuner_results',
            project_name='climate_model',
        )
        tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = build_model(best_hps)
        nn = Predictions(X_train, X_test, y_train, y_test)
        nn.model = best_model
        nn.train()
        nn.evaluate()

    
    
if __name__ == "__main__":
    main()




''' 
    tuner = kt.Hyperband(
    build_model,  
    objective='val_mae',
    max_epochs=45,
    directory='tuner_results',
    project_name='apartment_model',
    )
    
    tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps.values}")'''
    # Use the NeuralNetwork class for training and evaluation
