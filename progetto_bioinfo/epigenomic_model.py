from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

def FFNN_epi(input_shape):
  ffnn = Sequential([
    Input(shape=(input_shape,)),
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

def MLP_epi(input_shape):
  mlp = Sequential([
    Input(shape=(input_shape,)),
    Dense(1000, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
], "MLP")


  mlp.compile(
      optimizer="nadam",
      loss="binary_crossentropy",
      metrics=[
          "accuracy",
          AUC(curve="ROC", name="auroc"),
          AUC(curve="PR", name="auprc")
      ]
  )

  mlp.summary()
  return mlp