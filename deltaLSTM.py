import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print('Format: python3 deltaLSTM.py [Filename]')
    sys.exit(0)
  inputFile = str(sys.argv[1])
  df = pd.read_csv(inputFile)


  df = df[['TB', 'addr']]

  df = df.loc[df['TB'] == 0.0]

  addresses = df[['addr']].values

  #print(addresses)

  deltas = np.subtract(addresses[1:], addresses[:-1])

  #print(deltas)

  #print(np.max(deltas), np.min(deltas))

  nonOne = np.delete(deltas, np.where(deltas == 1.0))

  #print(nonOne, len(nonOne))

  scaler = MinMaxScaler(feature_range=(0, 1))
  normalizedDeltas = scaler.fit_transform(deltas)

  #plt.plot(np.arange(len(deltas)), deltas, marker='.', markersize = 6.0)
  #plt.show()


  # split into train and test sets
  train_size = int(len(normalizedDeltas) * 0.9)
  test_size = len(normalizedDeltas) - train_size
  train, test = normalizedDeltas[0:train_size,:], normalizedDeltas[train_size:len(normalizedDeltas),:]
  print(len(train), len(test))


  # reshape into X=t and Y=t+1
  look_back = 1
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)

  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  model = Sequential()
  model.add(LSTM(4, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

  #make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])

  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))

  # shift train predictions for plotting
  trainPredictPlot = np.empty_like(normalizedDeltas)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
  # shift test predictions for plotting
  testPredictPlot = np.empty_like(normalizedDeltas)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(trainPredict)+(look_back*2)+1:len(normalizedDeltas)-1, :] = testPredict
  # plot baseline and predictions
  plt.plot(scaler.inverse_transform(normalizedDeltas))
  plt.plot(trainPredictPlot)
  plt.plot(testPredictPlot)
  plt.show()