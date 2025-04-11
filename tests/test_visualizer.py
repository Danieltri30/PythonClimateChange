import unittest
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt
from src.visualizer import VisualizeData

class TestVisualizeData(unittest.TestCase):
    def setUp(self):
        # Dummy data for testing
        self.df = pd.DataFrame({
            # Using date strings; the visualizer methods extract the year via slicing.
            "time": ["2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01"],
            "co2_ppm": [400, 405, 410, 415],
            "temperature": [10, 15, 20, 25]
        })
        self.vis = VisualizeData()

    @patch("matplotlib.pyplot.show")
    def test_co2_over_time(self, mock_show):
        # Call the function; it should run without error and eventually call plt.show().
        self.vis.co2_over_time(self.df)
        mock_show.assert_called_once()
        # Optionally, verify a figure has been created.
        self.assertGreaterEqual(len(plt.get_fignums()), 1)
        plt.close('all')  # Close all figures

    @patch("matplotlib.pyplot.show")
    def test_temperature_levels_over_time(self, mock_show):
        self.vis.temperature_levels_over_time(self.df)
        mock_show.assert_called_once()
        self.assertGreaterEqual(len(plt.get_fignums()), 1)
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_co2_vs_Temperature(self, mock_show):
        self.vis.co2_vs_Temperature(self.df)
        mock_show.assert_called_once()
        self.assertGreaterEqual(len(plt.get_fignums()), 1)
        plt.close('all')

if __name__ == "__main__":
    unittest.main()
