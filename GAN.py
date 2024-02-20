import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Reshape, Conv1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('dataset.csv')

data = df[['Total', 'EV Penetration (in %)', 'GDP per capita (in billion US dollars)']].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

date_values = pd.to_datetime(df['Date']).values
date_values_scaled = MinMaxScaler().fit_transform(date_values.reshape(-1, 1))

data_combined = np.concatenate([data_scaled, date_values_scaled], axis=1)


def build_generator(latent_dim, n_features):
    model = Sequential()
    model.add(Dense(50, input_dim=latent_dim, activation='relu'))
    model.add(Reshape((50, 1)))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(n_features, activation='tanh'))
    return model


def build_discriminator(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

latent_dim = 100
n_features = data_combined.shape[1]

generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator((data_combined.shape[1], 1))

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

epochs = 200
batch_size = 32

for epoch in range(epochs):
    idx = np.random.randint(0, data_combined.shape[0], batch_size)
    real_samples = data_combined[idx]
    labels_real = np.ones((batch_size, 1))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_samples = generator.predict(noise)
    labels_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_samples, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

num_samples = 10
noise = np.random.normal(0, 1, (num_samples, latent_dim))
generated_data = generator.predict(noise)
generated_data = scaler.inverse_transform(generated_data[:, :3])  # Exclude the date values for display

columns = ['Total', 'EV Penetration (in %)', 'GDP per capita (in billion US dollars)']
generated_df = pd.DataFrame(generated_data, columns=columns)

print("Generated Data:")
print(generated_df)

gan.save('Model/gan_model_1.keras')