import pickle
import pandas as pd
# loading dataset
with open('dataset.pkl', 'rb') as f:
  dataset = pickle.load(f)

x_wind_speed = dataset['train'][:, 0, 0]  # extract the city index and temperature index
x_wind_direction = dataset['train'][:, 0, 1]
x_temperature = dataset['train'][:, 0, 2]  # extract the city index and temperature index
x_dew_point = dataset['train'][:, 0, 3]
x_air_pressure = dataset['train'][:, 0, 4]
x_rain_amount = dataset['train'][:, 0, 5]
# print("The shape of train set {0}".format(x1.shape))
# TRAIN set: building the dataframe requested for the Prophet model: columns and data

df = pd.DataFrame(data=x_wind_speed, columns=['WIND_SPEED'])
df['WIND_DIRECTION'] = x_wind_direction
df['TEMPERATURE'] = x_temperature
df['DREW_POINT'] = x_dew_point
df['AIR_PRESSURE'] = x_air_pressure
df['RAIN_AMOUNT'] = x_rain_amount
df['DATE'] = pd.date_range('01/01/2011 00:00:00', periods=len(x_wind_speed), freq='1H')
df['DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y %H:%M:%S')
order = ['DATE', 'WIND_SPEED', 'WIND_DIRECTION', 'TEMPERATURE', 'DREW_POINT', 'AIR_PRESSURE', 'RAIN_AMOUNT']
df = df[order]
df.to_csv('train.csv', index=False)

# x_wind_speed = dataset['test'][:, 0, 0]  # extract the city index and temperature index
# x_wind_direction = dataset['test'][:, 0, 1]
# x_temperature = dataset['test'][:, 0, 2]  # extract the city index and temperature index
# x_dew_point = dataset['test'][:, 0, 3]
# x_air_pressure = dataset['test'][:, 0, 4]
# x_rain_amount = dataset['test'][:, 0, 5]
# # print("The shape of train set {0}".format(x1.shape))
# # TRAIN set: building the dataframe requested for the Prophet model: columns and data
# df = pd.DataFrame(data=x_wind_speed, columns=['WIND_SPEED'])
# df['WIND_DIRECTION'] = x_wind_direction
# df['TEMPERATURE'] = x_temperature
# df['DREW_POINT'] = x_dew_point
# df['AIR_PRESSURE'] = x_air_pressure
# df['RAIN_AMOUNT'] = x_rain_amount
# df['DATE'] = pd.date_range('01/01/2019 00:00:00', periods=len(x_wind_speed), freq='1H')
# df['DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y %H:%M:%S')
# order = ['DATE', 'WIND_SPEED', 'WIND_DIRECTION', 'TEMPERATURE', 'DREW_POINT', 'AIR_PRESSURE', 'RAIN_AMOUNT']
# df = df[order]
# df.to_csv('test.csv', index=False)

