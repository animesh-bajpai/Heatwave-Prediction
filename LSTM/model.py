import numpy as np
from math import sqrt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_model(train, n_steps_in, n_steps_out, n_neurons=128, dropout_rate=0.3):
    n_features = train.shape[2]
    model = Sequential()
    model.add(LSTM(n_neurons, activation="tanh", input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(Dense(n_neurons, activation="tanh"))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(n_neurons, activation="tanh", return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation="linear")))
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model, X_train, Y_train, X_test, Y_test, n_epochs=120, batch_size=32):
    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    history = model.fit(
        X_train,
        Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        verbose=1,
        callbacks=[early_stop],
    )
    return history


def evaluate_model(model, scaler, X_test, Y_test):
    n_features = X_test.shape[2]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    Y_predict = model.predict(X_test)
    temp = np.zeros((X_test.shape[0], n_features))
    temp[:, 0] = Y_predict[:, 0].flatten()
    Y_predict = scaler.inverse_transform(temp)[:, 0]
    Y_predict = np.array(Y_predict)

    temp = np.zeros((Y_test.shape[0], n_features))
    for i in range(Y_test.shape[0]):
        temp[i][0] = Y_test[i][0]
    Y_test = scaler.inverse_transform(temp)
    Y_test = Y_test[:, 0]
    
    rmse = sqrt(mean_squared_error(Y_predict, Y_test))
    mae = mean_absolute_error(Y_predict, Y_test)
    return rmse, mae, Y_predict, Y_test
