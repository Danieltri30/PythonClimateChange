
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import base64
import datetime

import src.visualizergui as vm
#
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

class TestVisualizeGUI(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataFrame for CO2 and temperature plots.
        self.df_co2 = pd.DataFrame({
            'time': ['2000', '2001', '2002', '2003'],
            'co2_ppm': [400, 405, 410, 415],
            'temperature': [15, 15.5, 16, 16.5]
        })
        # Create a dummy DataFrame for predicted temperatures (using datetime objects).
        self.df_pred = pd.DataFrame({
            'time': pd.to_datetime(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01']),
            'predicted_temperature': [15, 15.2, 15.4, 15.6]
        })
        # Dummy DataFrame for clustering visualization.
        self.df_cluster = pd.DataFrame({
            'Longitude': np.random.uniform(-180, 180, 10),
            'Latitude': np.random.uniform(-90, 90, 10),
            'Cluster': np.random.randint(0, 5, 10)
        })
        # Dummy DataFrame for CO2 vs Temperature plot.
        self.df_temp_co2 = pd.DataFrame({
            'temperature': [15, 16, 17, 18],
            'co2_ppm': [400, 410, 420, 430]
        })

    def test_co2_over_time_for_gui(self):
        vis = vm.VisualizeData()
        result = vis.co2_over_time_for_gui(self.df_co2)
        # Verify that the function returns a non-empty Base64 string.
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # Optionally check that the Base64 string decodes correctly.
        try:
            decoded = base64.b64decode(result)
            self.assertTrue(len(decoded) > 0)
        except Exception:
            self.fail("co2_over_time_for_gui returned an invalid Base64 encoding.")

    def test_temperature_levels_over_time_for_gui(self):
        vis = vm.VisualizeData()
        result = vis.temperature_levels_over_time_for_gui(self.df_co2)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        try:
            decoded = base64.b64decode(result)
            self.assertTrue(len(decoded) > 0)
        except Exception:
            self.fail("temperature_levels_over_time_for_gui returned an invalid Base64 encoding.")

    def test_predicted_temperature_levels_over_time_for_gui(self):
        vis = vm.VisualizeData()
        result = vis.predicted_temperature_levels_over_time_for_gui(self.df_pred)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        try:
            decoded = base64.b64decode(result)
            self.assertTrue(len(decoded) > 0)
        except Exception:
            self.fail("predicted_temperature_levels_over_time_for_gui returned an invalid Base64 encoding.")

    def test_co2_vs_Temperature_for_gui(self):
        vis = vm.VisualizeData()
        result = vis.co2_vs_Temperature_for_gui(self.df_temp_co2)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_cluster_visualization_for_gui(self):
        vis = vm.VisualizeData()
        result = vis.cluster_visualization_for_gui(self.df_cluster, "Test")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_showelbow_for_gui(self):
        vis = vm.VisualizeData()
        # Provide a dummy list of distortions.
        distortions = [100, 80, 60, 50, 45, 43, 42, 41, 40, 39]
        result = vis.showelbow_for_gui(distortions)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_evaluate_prediction_model_for_gui(self):
        vis = vm.VisualizeData()
        dummy_history = DummyHistory()
        plots = vis.evaluate_prediction_model_for_gui(dummy_history)
        self.assertIn('mae_plot', plots)
        self.assertIn('r2_plot', plots)
        for key in plots:
            self.assertIsInstance(plots[key], str)
            self.assertTrue(len(plots[key]) > 0)

if __name__ == '__main__':
    unittest.main()
