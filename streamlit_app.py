import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load the pre-trained model
model = pickle.load(open("rnn.pkl", "rb"))

# Function to predict the next 7 days' load
def predict_next_7_days(data):
    # Preprocess the data
    data = data.dropna(subset=['loadweight'])
    train_load = data.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_load)

    # Reshape the data into 7 (day) time steps
    time_steps = 7
    X_train = []
    for i in range(time_steps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - time_steps:i, 0])

    X_train = np.array(X_train)

    # Reshaping X_train for RNN input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Select a random starting point for prediction
    start_idx = np.random.randint(0, len(X_train) - 1)
    last_sequence = X_train[start_idx]  # Taking a random sequence

    # Predict the next 7 days' load
    predicted = []  # Store predictions
    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, time_steps, 1))
        predicted.append(pred[0][-1])
        last_sequence = np.append(last_sequence[1:], pred[0][-1])

    # Rescale the predicted data
    predicted = sc.inverse_transform(np.array(predicted).reshape(-1, 1))

    return predicted

# Streamlit app
st.title("Load Prediction for the Next 7 Days")
st.write("Upload your CSV file and click the button to predict the next 7 days' load.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    if st.button("Predict Next 7 Days' Load"):
        predicted_values = predict_next_7_days(data)
        
        st.write("Predicted Load for the Next 7 Days:")
        for i in range(7):
            st.write(f"Day {i+1}: {predicted_values[i][0]:.2f} kgs")
else:
    st.write("Please upload a CSV file to proceed.")
