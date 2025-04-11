#pip install pandas
#pip install numpy
#pip install sklearn
#pip install tensorflow


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
import keras_tuner as kt
from tensorflow.keras.layers import Dropout
import data_processor as p
import visualizer as viz

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

class GeneralTasks:
    def run_prediction_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "FinalProcessedData.csv")
        X_train, X_test, y_train, y_test = p.load_and_prepare_data(csv_path)

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
    
    def run_cluster_simulation(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "CleanedGlobalLandTemp.csv")
        #Read in the dataset handled in data processor
        cluster_df = pd.read_csv(csv_path)

        #Call function from data processor to scale
        datautil = p.FineTuneClusterData()
        #WE will test with minmax scaling, and standard scaler
        scaled_features = datautil.minmaxnormalize_cluster_data(cluster_df)
        scaled_features_standard = datautil.standardnormalize_cluster_data(cluster_df)

        cutil = Clustering()
        #Get your K means function results
        kmeans_var = cutil.kmeans1()
        avg_city_df = cluster_df.copy()
        avg_city_df2 = cluster_df.copy()
        #Get prediction
        avg_city_df['Cluster'] = kmeans_var.fit_predict(scaled_features)
        avg_city_df2['Cluster'] = kmeans_var.fit_predict(scaled_features_standard)
        plotplacer = viz.VisualizeData()
        plotplacer.cluster_visualization(avg_city_df,"Min-Max")
        plotplacer.cluster_visualization(avg_city_df2,"Standard")
        #After various manual test we find out that k = 5 worked the best

        #After testing both Min-Max scaling and Standard Scaling, we
        #Found that standard scaling worked  the best 
        # Now lets use a loop to figure out best k
        klis = cutil.find_best_k(scaled_features) 
        plotplacer.showelbow(klis)

        

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "..", "data", "BestTrainedClusterData.csv")
        avg_city_df.to_csv(csv_path,index = False)   

class Clustering:
    #FOr testing multiple cluster values
    def kmeans1(self):
        return KMeans(n_clusters = 5,random_state=42)
    def kmeans2(self):
        return KMeans(n_clusters = 4,random_state=42)
    def kmeans3(self):
        return KMeans(n_clusters = 3,random_state=42)
    def kmeans4(self):
        return KMeans(n_clusters = 6,random_state=42)
    
    def find_best_k(self,sf):
        klis = []
        for k in range(1,11):
            km = KMeans(n_clusters = k,random_state=42)
            km.fit(sf)
            klis.append(km.inertia_)
        return klis    


# An old model I used in a previous project, its on my github
class NeuralNetwork:
    #Simple feed forward neural network using
    # Dense neurons and a relu activation function at the end
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
    
    #Below are just various implementations of the model I attempted to use
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
        self.model.save("T.keras")
        print("Model saved as 'T.h5'")

    def evaluate(self):
        loss, mae, r2 = self.model.evaluate(self.X_test, self.y_test)
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")


def main():
    print("Please Choose from the following options:\n1.Train ONLY the prediction model\n2.Train ONLY the clustering model\n3.Train both models.")
    inp = int(input())
    tool = GeneralTasks()
    if inp == 1:
        tool.run_prediction_model()   
    elif inp == 2:
        tool.run_cluster_simulation()
    elif inp == 3:
        x=2    

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

NewBestModel:
MAE : 0.0637
R²  : 0.6350

BestModel2:
MAE : 0.0642
R²  : 0.6379

'''    