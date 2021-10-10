import pandas as pd
from fbprophet import Prophet
import pickle
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import itertools

with open('dataset.pkl', 'rb') as f:
  dataset = pickle.load(f)

"""The shape of the data set."""
# dataset.keys()
# type(dataset)
# print(dataset['train'].shape)
# print(dataset['test'].shape)

"""extract the city index and temperature index"""
x1 = dataset['train'][:, 0, 0]
print("The shape of train set {0}".format(x1.shape))

"""building the dataframe"""
df = pd.DataFrame(data=x1, columns=['y'])
df['ds'] = pd.date_range('01/01/2000 00:00:00', periods=len(x1), freq='0.25H')
df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y %H:%M:%S')
print(df.head)

"""take a look at the data set"""
plt.plot(range(len(x1)), x1)
plt.show()

param_grid = {
'changepoint_prior_scale': [0.01, 0.1, 0.5], 'seasonality_prior_scale': [0.1, 1.0, 10.0], }
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

rmses = []

"""Fit model with given params"""

for params in all_params:
  m = Prophet(**params).fit(df)
  df_cv = cross_validation(m, horizon='30 days')
  df_p = performance_metrics(df_cv, rolling_window=1)
  rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)












