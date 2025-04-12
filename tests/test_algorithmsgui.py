import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import base64
import datetime

# Import the modules under test.
import algorithmsgui as am
import visualizergui as vm

# Dummy hyperparameter class for testing build_model functions.
class DummyHP:
    def Int(self, name, min_value, max_value, step):
        # Return the minimum value for simplicity.
        return min_value

    def Choice(self, name, choices):
        # Always choose the first option.
        return choices[0]

    def Boolean(self, name):
        # Return False so that optional layers are not added.
        return False

    def Float(self, name, min_value, max_value, step=None, sampling=None):
        # Return the minimum value.
        return min_value

# Dummy hyperparameter object that returns True for booleans
# and maximum values for integers/floats to force extra layers.
class DummyHPAllTrue:
    def Int(self, name, min_value, max_value, step):
        return max_value

    def Choice(self, name, choices):
        # Choose the last option
        return choices[-1]

    def Boolean(self, name):
        return True

    def Float(self, name, min_value, max_value, step=None, sampling=None):
        return max_value

# Dummy history object for simulating Keras training history.
class DummyHistory:
    def __init__(self):
        self.history = {
            'mae': [0.1, 0.05],
            'val_mae': [0.12, 0.06],
            'r2_metric': [0.8, 0.85],
            'val_r2_metric': [0.78, 0.83]
        }

class TestAlgorithmGUI(unittest.TestCase):

    def test_r2_metric(self):
        # Test that r2_metric returns 1.0 for perfect predictions.
        y_true = tf.constant([1.0, 2.0, 3.0])
        y_pred = tf.constant([1.0, 2.0, 3.0])
        r2 = am.r2_metric(y_true, y_pred)
        self.assertAlmostEqual(r2.numpy(), 1.0, places=5)

        # Test a non-perfect scenario.
        y_pred_diff = tf.constant([2.0, 2.0, 2.0])
        r2_diff = am.r2_metric(y_true, y_pred_diff)
        self.assertTrue(r2_diff.numpy() < 1.0)

    def test_build_model(self):
        # Using DummyHP, build_model should create a two-layer model.
        dummy_hp = DummyHP()
        model = am.build_model(dummy_hp)
        # Expect one hidden Dense and one output Dense layer.
        self.assertEqual(len(model.layers), 2)
        # Check that the optimizer is Adam.
        optimizer_name = model.optimizer.get_config()['name']
        self.assertEqual(optimizer_name, 'adam')

    def test_build_model2_default(self):
        # With DummyHP (booleans False), build_model2 should only add one hidden layer.
        dummy_hp = DummyHP()
        model = am.build_model2(dummy_hp)
        # When no extra layers or dropout are enabled, expect:
        # 1 hidden Dense and 1 output Dense layer.
        self.assertEqual(len(model.layers), 2)

    def test_build_model2_with_all_true(self):
        # With DummyHPAllTrue (booleans True), build_model2 should add more layers.
        dummy_hp = DummyHPAllTrue()
        model = am.build_model2(dummy_hp)
        # Expected layers:
        # - First Dense layer
        # - Second Dense layer (if use_second_layer True)
        # - Third Dense layer (if use_third_layer True)
        # - Dropout layer (if use_dropout True)
        # - Output Dense layer
        self.assertEqual(len(model.layers), 5)

    def test_clustering_find_best_k(self):
        # Create dummy data as an array with shape (100, 3).
        dummy_data = np.random.rand(100, 3)
        clustering = am.Clustering()
        best_k_list = clustering.find_best_k(dummy_data)
        # Expect a list of 10 values (for k=1 through 10).
        self.assertEqual(len(best_k_list), 10)
        for inertia in best_k_list:
            self.assertIsInstance(inertia, float)
            self.assertGreaterEqual(inertia, 0.0)

    def test_neural_network_model_creation(self):
        # Create a small dummy dataset.
        X_train = np.random.rand(10, 5)
        X_test = np.random.rand(5, 5)
        y_train = np.random.rand(10)
        y_test = np.random.rand(5)
        nn = am.NeuralNetwork(X_train, X_test, y_train, y_test)
        dummy_hp = DummyHP()
        # Build a neural network model using one of the defined methods.
        nn.d_model(dummy_hp)
        self.assertIsNotNone(nn.model)
        # Verify that the model has been compiled (has an optimizer attribute).
        self.assertTrue(hasattr(nn.model, 'optimizer'))

if __name__ == '__main__':
    unittest.main()
