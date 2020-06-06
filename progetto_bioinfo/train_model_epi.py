import pandas as pd
import numpy as np
import os
import compress_json

from typing import Tuple
from tqdm.auto import tqdm
from plot_keras_history import plot_history
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from epigenomic_dataset import load_epigenomes
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from sanitize_ml_labels import sanitize_ml_labels


def report(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    integer_metrics = accuracy_score, balanced_accuracy_score
    float_metrics = roc_auc_score, average_precision_score
    results1 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, np.round(y_pred))
        for metric in integer_metrics
    }
    results2 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in float_metrics
    }
    return {
        **results1,
        **results2
    }


def precomputed(results, model: str, holdout: int) -> bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
        (df.model == model) &
        (df.holdout == holdout)
    ).any()


def train_model_epi(models, epigenomes, nlabels, region_type, cell_line):
    # Reprod
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)

    y = nlabels[region_type].values.ravel()
    X = epigenomes[region_type]
    print("Num feature: " + str(X.shape[1]))
    splits = 51
    holdouts = StratifiedShuffleSplit(
        n_splits=splits, test_size=0.2, random_state=42)
    class_w = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_w = dict(enumerate(class_w))
    print("Class weights: " + str(class_w))
    
    if os.path.exists( cell_line + "_" + region_type + "_epigenomic.json"):
        results = compress_json.local_load( cell_line + "_" + region_type + "_epigenomic.json")
    else:
        results = []

    for i, (train, test) in tqdm(enumerate(holdouts.split(X, y)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            model_name = (
                model.__class__.__name__
                if model.__class__.__name__ != "Sequential"
                else model.name
            )
            if precomputed(results, model_name, i):
                continue

            model.fit(
                X[train],
                y[train],
                epochs=1000,
                shuffle=True,
                verbose=False,
                validation_split=0.1,
                batch_size=1024,
                class_weight=class_w,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min",
                                  patience=50, restore_best_weights=True),
                ]
            )
            results.append({
                "model": model_name,
                "run_type": "train",
                "holdout": i,
                **report(y[train], model.predict(X[train]))
            })
            results.append({
                "model": model_name,
                "run_type": "test",
                "holdout": i,
                **report(y[test], model.predict(X[test]))
            })
            compress_json.local_dump(
                results, cell_line + "_" + region_type + "_epigenomic.json")
            df = pd.DataFrame(results)
            df = df.drop(columns=["holdout"])

    return df
