import yfinance as yf
import os

# 设置股票代码和日期范围
ticker_symbol = "9984.T"
start_date = '2018-01-01'
end_date = '2023-01-01'

# 保存到CSV文件
csv_file = 'softbank_stock_data.csv'

# 检查文件是否存在
if not os.path.exists(csv_file):
    # 下载股票数据
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # 保存到CSV文件
    data.to_csv(csv_file)
    print(f"Data downloaded and saved to {csv_file}")
else:
    print(f"{csv_file} already exists. No download needed.")



import pandas as pd

# 读取CSV文件
csv_file = 'softbank_stock_data.csv'
stock_data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

# 显示前几行数据
print(stock_data.head())


close_prices = stock_data['Close']



from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 规范化数据
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = scaler.fit_transform(np.array(close_prices).reshape(-1, 1))

# 创建数据集
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 使用过去100天的数据预测下一天的价格
time_step = 100
X, y = create_dataset(close_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

from keras.models import load_model

# 加载保存的模型
model = load_model('lstm_model.h5')



# 做出预测
train_predict = model.predict(X)

# 反转预测
train_predict = scaler.inverse_transform(train_predict)

# 计算性能指标
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y, train_predict))


import matplotlib.pyplot as plt

predicted_data = train_predict
actual_data = stock_data['Close'].values[-len(train_predict):]

# 确保预测数据和实际数据长度一致
dates = range(len(actual_data))  # 如果有具体日期可以用具体日期

plt.figure(figsize=(10, 5))
plt.plot(dates, actual_data, label='Actual Data', color='blue', marker='o')
plt.plot(dates, predicted_data, label='Predicted Data', color='red', linestyle='--')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

