import pandas as pd
import scipy.io as sio
import datetime
import pickle
from fbprophet import Prophet

# with open('dataset.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(type(data))
# print(data.keys())
# print(data['train'].shape)
#
# print('----')
# data3 = sio.loadmat('step1.mat')
# print(type(data3))
# print(data3.keys())
# print(data3['Xtr'].shape)

# data = pd.read_excel('test.xlsx', index_col=0)
# data.to_csv('test.csv', encoding='utf-8')
# data = pd.read_csv('test.csv')
# test = pd.read_csv('csv/step1_Xp_1_0_.csv')
# test = sio.loadmat('data.mat')
# # print(data.keys())
# # print(data.shape)
#
# # print(test['X'].shape)
# x = test['X'][:, 0, 0]
# # print(x.shape)
# ds = pd.date_range('20000101', periods=70128)
# df = pd.DataFrame(data=x, index=ds, columns=['y'])
# df['ds'] = pd.date_range('20000101', periods=70128)

# print(df)

# df.to_csv('test.csv', index=False, header=False)
import pickle

# loading dataset
with open('/Users/yangyimin/PycharmProjects/WeatherForecasting/W_F_FIrst_Part/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(type(dataset))
print(dataset.keys())
print(dataset['train'].shape)
