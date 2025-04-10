import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pandas.core.algorithms import mode
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create temp CSV file for "ByCountry dataset"
        self.temp_files = []
        self.country_data = """dt,Country,AverageTemperature
2000-01-01,CountryA,10
2000-01-02,CountryA,20
2000-01-03,CountryA,30
"""
        country_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_ByCountry.csv")
        country_temp.write(self.country_data)
        country_temp.close()
        self.country_file = country_temp.name
        self.temp_files.append(self.country_file)

        # Create temporary CSV file for "ByMajorCity dataset"
        self.city_data = """dt,Country,City,AverageTemperature
2000-01-01,CountryA,CityX,15
2000-01-02,CountryA,CityX,25
2000-01-03,CountryA,CityX,35
"""
        city_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_ByMajorCity.csv")
        city_temp.write(self.city_data)
        city_temp.close()
        self.city_file = city_temp.name
        self.temp_files.append(self.city_file)

        # Create a temporary CSV file for "Deviation dataset"
        self.deviation_data = """Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D
2000,0.1,0.2,0.15,0.1,0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.2,1.75
2001,0.2,0.25,0.2,0.15,0.1,0.05,0.1,0.15,0.2,0.25,0.3,0.25,2.1
"""
        deviation_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_Deviation.csv")
        deviation_temp.write(self.deviation_data)
        deviation_temp.close()
        self.deviation_file = deviation_temp.name
        self.temp_files.append(self.deviation_file)

    def tearDown(self):
        # Remove temporary files after each test
        for file in self.temp_files:
            os.remove(file)


    def test_load_data(self):
        """Test that load_data correctly reads the CSV into a DataFrame"""
        dp = DataProcessor(self.country_file)
        df = dp.load_data()
        # Check that all expected columns are present and there are 3 rows
        self.assertIn("dt", df.columns)
        self.assertIn("Country", df.columns)
        self.assertIn("AverageTemperature", df.columns)
        self.assertEqual(len(df), 3)

    def test_clean_data(self):
        """Test that clean_data drops rows with missing values and computes normalized temperature correctly"""
        # Create a temporary CSV with one missing AverageTemperature value 
        data_with_na = """dt,Country,AverageTemperature
2000-01-01,CountryA,10
2000-01-02,CountryA,
2000-01-03,CountryA,30"""
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_ByCountry.csv")
        tmp_file.write(data_with_na)
        tmp_file.close()

        dp = DataProcessor(tmp_file.name)
        df_clean = dp.clean_data()
        # Should drop the row with missing value so only 2 rows remain 
        self.assertEqual(len(df_clean), 2)
        # Check thatt the normalized temperature column was added 
        self.assertIn("AverageTemperatureNormalized", df_clean.columns)

        # For CountryA the min and max are 10 and 30, so normalization:
        # For 10: (10-10)/(30-10) = 0.0, for 30: (30-10)/(30-10) = 1.0
        normalized_values = df_clean["AverageTemperatureNormalized"].to_numpy()
        self.assertAlmostEqual(normalized_values[0], 0.0)
        self.assertAlmostEqual(normalized_values[1], 1.0)
        os.remove(tmp_file.name)

    def test_get_features_and_target(self):
        """Test that get_features_and_target splits the data correctly into features and target"""
        # For the "ByCountry" file: features should contain Country, year, month, and day
        dp = DataProcessor(self.country_file)
        features, target = dp.get_features_and_target()

        # Check the shape: 3 rows and 4 columns (Country, year, month, day)
        self.assertEqual(features.shape, (3,4))
        # Check that target matches the original AverageTemperature values: 10, 20, 30
        np.testing.assert_array_almost_equal(target, np.array([10, 20, 30]))

        # For the "Deviation" file: features should be a 1D array of Years and target a 2D array of temperature data
        dp_dev = DataProcessor(self.deviation_file)
        features_dev, target_dev = dp_dev.get_features_and_target()
        # Features (Year) should have 2 elements 
        self.assertEqual(features_dev.shape, (2,))
        # Target should have shape (2,13) according to the 13 columns from Jan to J-D
        self.assertEqual(target_dev.shape, (2,13))

if __name__ == "__main__":
    unittest.main()

