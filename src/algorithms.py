import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
import keras_tuner as kt
from tensorflow.keras.layers import Dropout

#Custom r^2 I used in a previous project 
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# Build model based on fire ah hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric]) 
    return model

# Used to Normalize complete dataset for our algorithm
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df['time_ordinal'] = (df['time'] - df['time'].min()).dt.days

    scaler_time = MinMaxScaler()
    scaler_temp = MinMaxScaler()
    scaler_co2 = MinMaxScaler()

    df['time_scaled'] = scaler_time.fit_transform(df[['time_ordinal']])
    df['temperature_scaled'] = scaler_temp.fit_transform(df[['temperature']])
    df['co2_scaled'] = scaler_co2.fit_transform(df[['co2_ppm']])

    X = df[['time_scaled', 'co2_scaled']]
    y = df['temperature_scaled']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# An old model I used in a previous project, its on my github
class NeuralNetwork:
    #Simple feed forward neural network using 
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
    
    def b_model(self, hp=None):
        self.model = Sequential()
        units = hp.Int('units', min_value=32, max_value=128, step=32) if hp else 64
        self.model.add(Dense(units=units, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric]) 
    
    def c_model(self, hp=None):
        self.model = Sequential()
        
        # First hidden layer (tuned or default units)
        units = hp.Int('units', min_value=32, max_value=256, step=32) if hp else 64
        self.model.add(Dense(units=units, activation='relu'))
        
        self.model.add(Dense(32, activation='relu'))  # You can also make this tunable if you want
        
        
        self.model.add(Dense(1, activation='linear'))
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae', r2_metric]
        )
    def d_model(self, hp=None):
        self.model = Sequential()

        units_1 = hp.Int('units_1', min_value=64, max_value=256, step=32) if hp else 128
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16) if hp else 64

        self.model.add(Dense(units=units_1, activation='relu'))
        self.model.add(Dense(units=units_2, activation='relu'))

        self.model.add(Dense(1, activation='linear'))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric])    

    def e_model(self, hp=None):
        self.model = Sequential()

        units_1 = hp.Int('units_1', min_value=64, max_value=256, step=32) if hp else 128
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16) if hp else 64
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.05) if hp else 0.1

        self.model.add(Dense(units=units_1, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=units_2, activation='relu'))

        self.model.add(Dense(1, activation='linear'))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric])    



    def train(self, epochs=50, batch_size=32):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size)
        self.model.save("bestd1000(1)_model.h5")
        print("Model saved as 'best_model.h5'")

    def evaluate(self):
        loss, mae, r2 = self.model.evaluate(self.X_test, self.y_test)
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data("data/FinalProcessedData.csv")

    tuner = kt.Hyperband(
        build_model,
        objective='val_mae',
        max_epochs=1000,
        directory='tuner_dense',
        project_name='dense_model_test'
    )

    tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps.values}")

    nn = NeuralNetwork(X_train, X_test, y_train, y_test)
    nn.d_model(hp=best_hps)
    nn.train(epochs=1000)
    nn.evaluate()

if __name__ == "__main__":
    main()


'''
bmodel:
MAE : 0.0699
R²  : 0.5555


cmodel: 
MAE : 0.0692
R²  : 0.5569

cmodel150epochs:
MAE : 0.0662
R²  : 0.6005

dmodel:
MAE : 0.0660
R²  : 0.6051

e50model:
MAE : 0.0685
R²  : 0.5652

e150model:
MAE : 0.0656
R²  : 0.5995

e150(1)model:
MAE : 0.0646
R²  : 0.6249

d1000model:
MAE : 0.0645
R²  : 0.6283

d1000(1)model:
MAE : 0.0637
R²  : 0.6327

'''    