from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization


def lstm_model(inputShape):
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(inputShape), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(LSTM(units=25))
    model.add(Dense(units=1, activation='sigmoid'))
    return model
