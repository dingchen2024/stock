#1. Gather Stock Price Data
#a. Using APIs
import requests
import pandas as pd

# Example using Alpha Vantage
API_KEY = 'your_api_key'
symbol = 'AAPL'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&datatype=csv'

response = requests.get(url)
with open('stock_data.csv', 'wb') as file:
    file.write(response.content)
    
df = pd.read_csv('stock_data.csv')
print(df.head())

#b. Using Web Scraping
import requests
from bs4 import BeautifulSoup

url = 'https://finance.yahoo.com/quote/AAPL/history?p=AAPL'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Parse HTML to extract data
# This is an example and might need adjustments based on the website structure
rows = soup.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text for col in cols]
    data.append(cols)

df = pd.DataFrame(data)
print(df.head())

import requests
from bs4 import BeautifulSoup

url = 'https://finance.yahoo.com/quote/AAPL/history?p=AAPL'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Parse HTML to extract data
# This is an example and might need adjustments based on the website structure
rows = soup.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text for col in cols]
    data.append(cols)

df = pd.DataFrame(data)
print(df.head())

#2. Stpre the Data
import sqlite3

# Connect to a database (or create one)
conn = sqlite3.connect('stock_data.db')
c = conn.cursor()

# Create a table
c.execute('''CREATE TABLE IF NOT EXISTS stock_prices
             (date text, open real, high real, low real, close real, volume integer)''')

# Insert data into the table
for index, row in df.iterrows():
    c.execute("INSERT INTO stock_prices VALUES (?,?,?,?,?,?)", tuple(row))

conn.commit()
conn.close()

#3. Preprocess the Data
from sklearn.preprocessing import MinMaxScaler

# Assuming 'df' is your DataFrame containing the stock data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']].values)

# Split the data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare the data for the LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#4. Develop the AI Model
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, batch_size=1, epochs=1)

# Evaluate the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

#5. Deploy the AI Model
# Function to make predictions with new data
def predict_stock_price(new_data):
    new_data_scaled = scaler.transform(new_data)
    X_new = []
    for i in range(look_back, len(new_data_scaled)):
        X_new.append(new_data_scaled[i-look_back:i, 0])
    X_new = np.array(X_new)
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
    predictions = model.predict(X_new)
    predictions = scaler.inverse_transform(predictions)
    return predictions
