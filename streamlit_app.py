import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import pickle

# Load the model using pickle
with open("rnn.pkl", "rb") as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(page_title="📈 Waste Load Predictor", layout="centered")

# Header Styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <div class="title">♻️ Waste Load Prediction Dashboard</div>
    <div class="subtitle">Upload your CSV file to predict waste load for the next 7 days</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a CSV file with a `loadweight` column", type="csv")

# Predict function
def predict_next_7_days(data):
    data = data.dropna(subset=['loadweight'])
    train_load = data.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_load)

    if len(training_set_scaled) < 8:
        st.error("🚫 Not enough data. Please upload at least 8 rows.")
        return

    time_steps = 7
    X_train = []
    for i in range(time_steps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - time_steps:i, 0])

    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    start_idx = np.random.randint(0, len(X_train) - 1)
    last_sequence = X_train[start_idx]

    predicted = []
    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, -1))  # assuming scikit-learn model
        predicted.append(pred[0])
        last_sequence = np.append(last_sequence[1:], pred)

    predicted = sc.inverse_transform(np.array(predicted).reshape(-1, 1))
    return predicted

# When file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

    if st.button("🔮 Predict Next 7 Days"):
        predicted_values = predict_next_7_days(data)

        if predicted_values is not None:
            st.success("✅ Prediction complete!")

            st.subheader("📈 Predicted Load for Next 7 Days")
            for i, val in enumerate(predicted_values, start=1):
                st.write(f"**Day {i}**: {val[0]:.2f} kgs")

            # Plot chart
            st.subheader("📉 Load Forecast Chart")
            days = [f"Day {i}" for i in range(1, 8)]
            plt.figure(figsize=(10, 4))
            plt.plot(days, predicted_values, marker='o', linestyle='-', color='#1F77B4')
            plt.title("Next 7 Days Predicted Waste Load")
            plt.xlabel("Day")
            plt.ylabel("Load (kgs)")
            plt.grid(True)
            st.pyplot(plt)
else:
    st.info("📁 Please upload a CSV file to begin.")
