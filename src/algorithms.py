#import

def build_model(hp):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(365, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae', r2_metric]
    )
    return model

# NeuralNetwork Class
class NeuralNetwork:
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