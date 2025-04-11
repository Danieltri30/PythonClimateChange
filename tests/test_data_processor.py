import unittest
import pandas as pd
import tempfile
import os
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_files = []
        # Temporary CSV for a country-like dataset (using "time" column)
        self.country_data = """time,Country,AverageTemperature
2000-01-01,CountryA,10
2000-01-02,CountryA,20
2000-01-03,CountryA,30
"""
        country_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_ByCountry.csv")
        country_temp.write(self.country_data)
        country_temp.close()
        self.country_file = country_temp.name
        self.temp_files.append(self.country_file)

        # Temporary CSV for a city-like dataset (using "time" column)
        self.city_data = """time,Country,City,AverageTemperature
2000-01-01,CountryA,CityX,15
2000-01-02,CountryA,CityX,25
2000-01-03,CountryA,CityX,35
"""
        city_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_ByMajorCity.csv")
        city_temp.write(self.city_data)
        city_temp.close()
        self.city_file = city_temp.name
        self.temp_files.append(self.city_file)

        # Temporary CSV for a global deviation dataset remains unchanged.
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
        for file in self.temp_files:
            os.remove(file)

    def test_load_data(self):
        # Test that load_data reads the CSV correctly.
        dp = DataProcessor(self.country_file)
        df = dp.load_data()
        # Expect the "time" column now instead of "dt"
        self.assertIn("time", df.columns)
        self.assertIn("Country", df.columns)
        self.assertIn("AverageTemperature", df.columns)
        self.assertEqual(len(df), 3)

    def test_merge_data(self):
        # Create two data frames with a "time" column.
        frame1 = pd.DataFrame({
            "time": ["2000-01-01", "2000-01-01", "2000-01-02"],
            "temperature": [10, 20, 30]
        })
        frame2 = pd.DataFrame({
            "time": ["2000-01-01", "2000-01-02", "2000-01-02"],
            "co2_ppm": [400, 405, 410]
        })
        dp = DataProcessor("dummy_path")
        merged_df = dp.merge_data(frame1, frame2)
        # After grouping by month, all dates fall into the same month.
        # We expect a merged DataFrame with one row and the columns: time, temperature, co2_ppm.
        self.assertFalse(merged_df.empty)
        self.assertIn("time", merged_df.columns)
        self.assertIn("temperature", merged_df.columns)
        self.assertIn("co2_ppm", merged_df.columns)
        self.assertEqual(len(merged_df), 1)

    def test_get_true_location_no_latlon(self):
        # Test get_true_location on data without 'LatDim' and 'LonDim'
        data = """time,temperature
2000-01-01,10
2000-01-02,20"""
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        tmp_file.write(data)
        tmp_file.close()
        dp = DataProcessor(tmp_file.name)
        # Since the required columns are not present, the method prints a message and returns the DataFrame.
        df = dp.get_true_location()
        self.assertIn("temperature", df.columns)
        os.remove(tmp_file.name)

if __name__ == "__main__":
    unittest.main()
