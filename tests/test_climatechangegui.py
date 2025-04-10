import unittest
import tkinter as tk
import pandas as pd
from unittest.mock import patch
from src.climatechangegui import ClimateChangeGUI
from src.data_processor import DataProcessor

# Define a dummy clean_data function that returns different DataFrames
# based on the file_path used in the DataProcessor instance.
def dummy_clean_data(self):
    if "GlobalLandTemperaturesByCountry" in self.file_path:
        # Create dummy data for country-level temperatures.
        return pd.DataFrame({
            "dt": ["2000-01-01", "2001-01-01"],
            "AverageTemperature": [10.0, 11.0],
            "Country": ["CountryA", "CountryA"]
        })
    elif "GlobalLandTemperaturesByMajorCity" in self.file_path:
        # Create dummy data for city-level temperatures.
        return pd.DataFrame({
            "dt": ["2000-01-01", "2001-01-01"],
            "AverageTemperature": [15.0, 16.0],
            "City": ["CityA", "CityA"],
            "Country": ["CountryA", "CountryA"]
        })
    elif "GlobalTemperatureDeviation" in self.file_path:
        # Create dummy data for global temperature deviation.
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

class TestClimateChangeGUI(unittest.TestCase):

    def setUp(self):
        # Patch DataProcessor.clean_data to use our dummy implementation
        patcher = patch.object(DataProcessor, "clean_data", new=dummy_clean_data)
        self.addCleanup(patcher.stop)
        patcher.start()
        # Create a root Tk window and hide it during tests
        self.root = tk.Tk()
        self.root.withdraw()
        self.gui = ClimateChangeGUI(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_load_data(self):
        # Ensure that the dataframes for country, city, and global data were created.
        self.assertIsInstance(self.gui.df_country, pd.DataFrame)
        self.assertIsInstance(self.gui.df_city, pd.DataFrame)
        self.assertIsInstance(self.gui.df_global, pd.DataFrame)
        
        # For country and city datasets, check that the "year" column is added.
        self.assertIn("year", self.gui.df_country.columns)
        self.assertIn("year", self.gui.df_city.columns)
        # For global data, check that the "Year" column is present.
        self.assertIn("Year", self.gui.df_global.columns)

    def test_update_mode(self):
        # Test country mode: the dropdown should list unique country names.
        self.gui.mode_var.set("country")
        self.gui.update_mode()
        expected_country_options = sorted(self.gui.df_country["Country"].unique().tolist())
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_country_options)

        # Test city mode: the dropdown should list unique city names.
        self.gui.mode_var.set("city")
        self.gui.update_mode()
        expected_city_options = sorted(self.gui.df_city["City"].unique().tolist())
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_city_options)

        # Test global mode: the dropdown should list the columns of the global dataframe (except the first column).
        self.gui.mode_var.set("global")
        self.gui.update_mode()
        expected_global_options = list(self.gui.df_global.columns[1:14])
        self.assertEqual(list(self.gui.selection_dropdown["values"]), expected_global_options)

    def test_plot_data(self):
        # Test plotting for country mode.
        self.gui.mode_var.set("country")
        self.gui.update_mode()
        self.gui.selection_var.set("CountryA")
        self.gui.plot_data()
        # Retrieve the plotted line(s) from the axes.
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        xdata = list(line.get_xdata())
        ydata = list(line.get_ydata())
        self.assertEqual(xdata, [2000, 2001])
        self.assertEqual(ydata, [10.0, 11.0])
        self.assertEqual(self.gui.ax.get_title(), "Average Temperature Over Time: CountryA")

        # Test plotting for city mode.
        self.gui.mode_var.set("city")
        self.gui.update_mode()
        self.gui.selection_var.set("CityA")
        self.gui.plot_data()
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        xdata = list(line.get_xdata())
        ydata = list(line.get_ydata())
        self.assertEqual(xdata, [2000, 2001])
        self.assertEqual(ydata, [15.0, 16.0])
        self.assertEqual(self.gui.ax.get_title(), "Average Temperature Over Time: CityA")

        # Test plotting for global mode.
        self.gui.mode_var.set("global")
        self.gui.update_mode()
        # The global mode options come from the global dataframe; choose "Jan" as an example.
        self.gui.selection_var.set("Jan")
        self.gui.plot_data()
        lines = self.gui.ax.get_lines()
        self.assertGreater(len(lines), 0)
        line = lines[0]
        xdata = list(line.get_xdata())
        ydata = list(line.get_ydata())
        self.assertEqual(xdata, [2000, 2001])
        self.assertEqual(ydata, [0.1, 0.2])
        self.assertEqual(self.gui.ax.get_title(), "Global Temperature Anomaly (Jan)")

if __name__ == "__main__":
    unittest.main()
