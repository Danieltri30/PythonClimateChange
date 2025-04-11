import unittest
import tkinter as tk
import pandas as pd
from unittest.mock import patch
from src.data_processor import DataProcessor

# Define a dummy "clean_data" function that returns dummy data.
def dummy_load_data(self):
    if "GlobalLandTemperaturesByCountry" in self.file_path:
        # Dummy country-level data.
        return pd.DataFrame({
            "dt": ["2000-01-01", "2001-01-01"],
            "AverageTemperature": [10.0, 11.0],
            "Country": ["CountryA", "CountryA"]
        })
    elif "GlobalLandTemperaturesByMajorCity" in self.file_path:
        # Dummy city-level data.
        return pd.DataFrame({
            "dt": ["2000-01-01", "2001-01-01"],
            "AverageTemperature": [15.0, 16.0],
            "City": ["CityA", "CityA"],
            "Country": ["CountryA", "CountryA"]
        })
    elif "GlobalTemperatureDeviation" in self.file_path:
        # Dummy global temperature anomaly data.
        return pd.DataFrame({
            "Year": [2000, 2001],
            "Jan": [0.1, 0.2],
            "Feb": [0.3, 0.4],
            "Mar": [0.5, 0.6],
            "Apr": [0.7, 0.8],
            "May": [0.9, 1.0],
            "Jun": [1.1, 1.2],
            "Jul": [1.3, 1.4],
            "Aug": [1.5, 1.6],
            "Sep": [1.7, 1.8],
            "Oct": [1.9, 2.0],
            "Nov": [2.1, 2.2],
            "Dec": [2.3, 2.4],
            "J-D": [2.5, 2.6]
        })
    else:
        return pd.DataFrame()

from src.climatechangegui import ClimateChangeGUI

class TestClimateChangeGUI(unittest.TestCase):

    def setUp(self):
        # Patch DataProcessor.clean_data in src.data_processor even though it does not exist,
        # using create=True to add the attribute.
        patcher = patch.object(DataProcessor, "clean_data", new=dummy_load_data, create=True)
        self.addCleanup(patcher.stop)
        patcher.start()

        # Create and hide a Tkinter root window.
        self.root = tk.Tk()
        self.root.withdraw()
        self.gui = ClimateChangeGUI(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_load_data(self):
        # Verify that the GUI loaded data and that date columns are added.
        self.assertIsInstance(self.gui.df_country, pd.DataFrame)
        self.assertIsInstance(self.gui.df_city, pd.DataFrame)
        self.assertIsInstance(self.gui.df_global, pd.DataFrame)
        self.assertIn("year", self.gui.df_country.columns)
        self.assertIn("year", self.gui.df_city.columns)
        self.assertIn("Year", self.gui.df_global.columns)

    def test_update_mode(self):
        # Test that the dropdown options are updated based on mode.

        # Country mode.
        self.gui.mode_var.set("country")
        self.gui.update_mode()
        expected_options = sorted(self.gui.df_country["Country"].unique().tolist())
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_options)

        # City mode.
        self.gui.mode_var.set("city")
        self.gui.update_mode()
        expected_options = sorted(self.gui.df_city["City"].unique().tolist())
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_options)

        # Global mode.
        self.gui.mode_var.set("global")
        self.gui.update_mode()
        expected_options = list(self.gui.df_global.columns[1:14])
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_options)

    def test_plot_data(self):
        # Test plotting for each mode.

        # Country mode.
        self.gui.mode_var.set("country")
        self.gui.update_mode()
        self.gui.selection_var.set("CountryA")
        self.gui.plot_data()
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        self.assertEqual(list(line.get_xdata()), [2000, 2001])
        self.assertEqual(list(line.get_ydata()), [10.0, 11.0])
        self.assertEqual(self.gui.ax.get_title(), "Average Temperature Over Time: CountryA")

        # City mode.
        self.gui.mode_var.set("city")
        self.gui.update_mode()
        self.gui.selection_var.set("CityA")
        self.gui.plot_data()
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        self.assertEqual(list(line.get_xdata()), [2000, 2001])
        self.assertEqual(list(line.get_ydata()), [15.0, 16.0])
        self.assertEqual(self.gui.ax.get_title(), "Average Temperature Over Time: CityA")

        # Global mode.
        self.gui.mode_var.set("global")
        self.gui.update_mode()
        self.gui.selection_var.set("Jan")
        self.gui.plot_data()
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        self.assertEqual(list(line.get_xdata()), [2000, 2001])
        self.assertEqual(list(line.get_ydata()), [0.1, 0.2])
        self.assertEqual(self.gui.ax.get_title(), "Global Temperature Anomaly (Jan)")

if __name__ == "__main__":
    unittest.main()
