from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Reshape, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import LSTM

def CNN():
  cnn = Sequential([
    Input(shape=(200, 4)),
    Reshape((200, 4, 1)),
    Conv2D(128, kernel_size=(10, 2), activation="relu"),
    Conv2D(128, kernel_size=(10, 2), activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Conv2D(64, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
    Conv2D(32, kernel_size=(10, 1), activation="relu"),
    Conv2D(32, kernel_size=(10, 1), activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(16, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="sigmoid")
  ], "CNN")
  
  cnn.compile(
      optimizer="nadam",
      loss="binary_crossentropy",
      metrics=[
          "accuracy",
          AUC(curve="ROC", name="auroc"),
          AUC(curve="PR", name="auprc")
      ]
  )

  cnn.summary()
  return cnn

def FFNN():
  ffnn = Sequential([
    Input(shape=(200, 4)),
    Flatten(),
    Dense(1024, activation="relu"),
    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dropout(0.4),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
], "FFNN")

  ffnn.compile(
      optimizer="nadam",
      loss="binary_crossentropy",
      metrics=[
          "accuracy",
          AUC(curve="ROC", name="auroc"),
          AUC(curve="PR", name="auprc")
      ]
  )

  ffnn.summary()
  return ffnn