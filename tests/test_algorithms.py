import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import src.algorithms as alg
from src.algorithms import NeuralNetwork, build_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Assign Keras symbols and dummy r2_metric to the algorithms module.
        alg.Sequential = Sequential
        alg.Dense = Dense
        alg.r2_metric = lambda y_true, y_pred: 0.0

        # Create dummy data with appropriate shapes.
        self.X_train = np.random.rand(2, 365, 1)
        self.X_test = np.random.rand(2, 365, 1)
        self.y_train = np.random.rand(2)
        self.y_test = np.random.rand(2)
        self.nn = NeuralNetwork(self.X_train, self.X_test, self.y_train, self.y_test)

    def test_build_model(self):
        # Instead of calling the removed build_model(), use one of the available model builders.
        self.nn.d_model()  # Using d_model as the default model builder.
        self.assertIsNotNone(self.nn.model)
        # Check that the model is compiled (has an optimizer attribute).
        self.assertTrue(hasattr(self.nn.model, "optimizer"))

    def test_train(self):
        self.nn.d_model()  # Use the updated model builder.
        # Patch the fit method to avoid actual training.
        self.nn.model.fit = MagicMock()

        # Patch the print function to capture the output.
        with patch("builtins.print") as mock_print:
            self.nn.train(epochs=1, batch_size=1)
        
        self.nn.model.fit.assert_called_once()
        # Instead of checking a call to self.model.save(),
        # check that the expected message is printed.
        mock_print.assert_called_with("Model saved as 'best_model.h5'")

    def test_evaluate(self):
        self.nn.d_model()  # Use the updated model builder.
        # Simulate model evaluation return values.
        self.nn.model.evaluate = MagicMock(return_value=[0.5, 0.1, 0.9])
        
        printed = []
        def fake_print(*args, **kwargs):
            printed.append(" ".join(str(a) for a in args))
        
        with patch("builtins.print", new=fake_print):
            self.nn.evaluate()
            
        # Verify that the expected formatted strings are printed.
        self.assertIn("MAE : 0.1000", printed)
        self.assertIn("R²  : 0.9000", printed)

if __name__ == "__main__":
    unittest.main()
