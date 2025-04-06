import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st
yfin.pdr_override()
start = '2015-09-01'
end = '2023-03-02'

st.title('Cryptocurrency Price Prediction')

user_input = st.text_input('Enter Crypto', 'LTC-USD')
df = pdr.get_data_yahoo(user_input, start, end)
st.subheader('Price V/S Day')
fig = plt.figure(figsize=(20,10))
continent_list = list(['Open','Close','Volume','High','Low'])
user_input_2 = st.selectbox(label = "Choose a Y-Column", options = continent_list)
#user_input_2 = (st.text_input('Choose the Y- Column','Close', key = "graph"))
user_input_3 = (int)(st.text_input('Choose the length X- Column',10, key = "length"))
plt.plot(df.index[::][:user_input_3],df[user_input_2][::][:user_input_3])
plt.xlabel("Days")  
plt.ylabel("Price of Litecoin in Dollars") 
plt.title("Litecoin line plot of Date vs Price")  
st.pyplot(fig)
st.subheader('Data from 2015 - 2023')
st.write(df.describe())

st.subheader('Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Moving Averages')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
l1 = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r', label = 'Moving averages 100 days')
plt.plot(ma200,'g', label = 'moving averages of 200 days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
st.pyplot(l1)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_arr.shape[0]):
  x_train.append(data_training_arr[i-100: i])
  y_train.append(data_training_arr[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

model = load_model('modellstm.h5')

past_100 = data_training.tail(100)
final_df = past_100.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test =[]
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor

st.subheader('Predicted vs Actual')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted , 'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()
st.pyplot(fig2)
