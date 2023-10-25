## Stock_Price_prediction

#Loading and Visualizing Data:

The code begins by importing necessary libraries and loading historical stock price data for Apple Inc. (AAPL) from a CSV file named 'AAPL.csv.'
It uses the pandas library to load and preprocess the data.
The historical closing prices are visualized using matplotlib.

#Data Preprocessing:

The closing price data is extracted and normalized using the Min-Max scaling technique to bring the values between 0 and 1. This normalization helps the neural network to converge more effectively.

#Data Splitting:

The data is divided into training and testing sets, with 80% of the data used for training and the remaining 20% for testing.

#Creating Sequences for LSTM:

A function create_sequences is defined to create sequences for training the LSTM model. It takes a sequence length (in this case, 60) and generates input-output pairs.

#LSTM Model Building:

A Sequential model is created using Keras with two LSTM layers, followed by two dense (fully connected) layers.
The model architecture is defined to learn and predict stock prices based on the historical sequences.

#Model Compilation:

The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function, which is common for regression problems like stock price prediction.

#Model Training:

The model is trained using the training data, and the training process is displayed with the specified batch size and number of epochs.

#Making Predictions:

The model is used to make predictions on the test data.

#Calculating RMSE (Root Mean Squared Error):

The Root Mean Squared Error is calculated to assess the model's accuracy in predicting stock prices. Lower RMSE values indicate better predictions.

#Visualization:

The code generates a plot to visualize the training data, the actual stock prices (in the testing data), and the model's predictions. The legend shows the differentiation between the training data, actual testing data, and predicted values.
The code demonstrates a basic workflow for training an LSTM model for stock price prediction. Keep in mind that real-world stock price prediction is a complex task, and this example is a starting point. More sophisticated models and additional features are often required to improve prediction accuracy.




