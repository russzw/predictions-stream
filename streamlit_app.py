import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle

# keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Load the pre-trained model
model = pickle.load(open("rnn.pkl", "rb"))

# Function to predict the next 7 days' load
def predict_next_7_days(data):
    # Preprocess the data
    data = data.dropna(subset=['loadweight'])
    train_load = data.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_load)

    # Reshape the data into 30 (day) time steps
    X_train = []
    y_train = []
    time_steps = 7
    for i in range(time_steps, len(training_set_scaled) - time_steps):
        X_train.append(training_set_scaled[i - time_steps:i, 0])
        y_train.append(training_set_scaled[i:i + time_steps, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for RNN input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Predict the next 7 days' load
    predicted_load = model.predict(X_train[-1:])

    # Inverse scale the predictions
    predicted_load = sc.inverse_transform(predicted_load)

    # Example of predicting the next 24 hours
    last_sequence = X_train[-1]  # Taking the last sequence from validation set
    predicted = []  # Store predictions
    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, time_steps, 1))
        predicted.append(pred[0][-1])
        last_sequence = np.append(last_sequence[1:], pred[0][-1])

    # Rescale the predicted data
    predicted = sc.inverse_transform(np.array(predicted).reshape(-1, 1))

    return predicted

# Get user input
data = pd.read_csv("hdt.csv")

# Predict the next 7 days' load
predicted_values = predict_next_7_days(data)

# Print the predicted values for the next 7 Day
st.write("Predicted Load for the Next Days:")

for i in range(7):
    st.write(f"Day {i+1}: {predicted_values[i][0]:.2f} kgs")