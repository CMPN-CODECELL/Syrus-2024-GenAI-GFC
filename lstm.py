import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your dataset
# Assuming your dataset is in a CSV file named 'ev_data.csv'
df = pd.read_csv('ev_sales_data.csv')

data = df['Total'].values.reshape(-1, 1)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define sequence length (adjust as needed)
sequence_length = 10

# Create sequences for training the model
sequences, labels = create_sequences(data_normalized, sequence_length)

# Split data into training and testing sets
split = int(0.8 * len(sequences))
train_sequences, train_labels = sequences[:split], labels[:split]
test_sequences, test_labels = sequences[split:], labels[split:]

X_train, y_train = train_sequences[:, :-1], train_sequences[:, -1]
X_test, y_test = test_sequences[:, :-1], test_sequences[:, -1]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get the actual sales values
predictions_actual = scaler.inverse_transform(predictions)

# You can now use the model for future predictions
# Prepare the input sequence for prediction based on the latest data
latest_data = data_normalized[-sequence_length:]
latest_data = latest_data.reshape((1, sequence_length, 1))

# Make prediction for the future
future_prediction = model.predict(latest_data)

# Inverse transform the future prediction to get the actual sales value
future_prediction_actual = scaler.inverse_transform(future_prediction)

print(f'Predicted Sales for the Future: {future_prediction_actual[0, 0]}')