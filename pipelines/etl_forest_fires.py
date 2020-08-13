import pandas as pd
import numpy as np

from dummy_transformer import dummify

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# categorical_cols = ['month', 'day']
# numerical_cols = [col for col in df.columns.tolist() if not col in categorical_cols]

# ### Forest Weather Index (FWI)
# Is a Canadian system for rating fire danter. It includes six components (Cortez et al).
# - Fine Fuel Moisture Code (FFMC): moisture content surface litter and influences ignition and fire spread
# - Initial Spread Index (ISI): Fire velocity spread
# - Duff Moisture Code (DMC): moisture content of shallow organic layers
# - Drought Code (DC): moisture content of deep organic layer

# ### Weather Indicators
# - temp: Temperature (Celcius)
# - RH: Relative humidity (%)
# - wind (km/h)
# - rain (mm/m2

# Remove Rain Outliers
# sorted_rain = df.rain.sort_values(ascending=False)
# print(sorted_rain.head(10))
# print(sorted_rain.tail(5))
# df = df[df.rain < 6.0]

# ### Target: Burnt Are
# Remove Outliers
# max_area = 70
# df = df[df.area < max_area]

if __name__ == "__main__":
    df = load_data('lake/forest_fires/forestfires.csv')
    df = dummify(df, 'month', prefix='month')
    df = dummify(df, 'day', prefix='day')
