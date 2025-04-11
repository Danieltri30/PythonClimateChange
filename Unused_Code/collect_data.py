if __name__ == "__main__":
    import requests
    import pandas as pd


    latitude = 40.7128
    longitude = -74.0060

    start_date = "2022-01-01"
    end_date = "2022-12-31"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=auto"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['daily'])
    df.head()

    df.to_csv("Temperature.csv",index = False)
