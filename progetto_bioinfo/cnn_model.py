import pandas as pd
import numpy as np
from typing import Tuple
import os
import compress_json
from tqdm.auto import tqdm
from plot_keras_history import plot_history
from barplots import barplots

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

def CNN():
  cnn = Sequential([
    Input(shape=(200, 4)),
    Reshape((200, 4, 1)),
    Conv2D(64, kernel_size=(10, 2), activation="relu"),
    Conv2D(64, kernel_size=(10, 2), activation="relu"),
    Dropout(0.3),
    Conv2D(32, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
    Conv2D(32, kernel_size=(10, 1), activation="relu"),
    Conv2D(32, kernel_size=(10, 1), activation="relu"),
    Dropout(0.3),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
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
