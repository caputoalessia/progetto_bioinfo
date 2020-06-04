import pandas as pd
import numpy as np
import os
import compress_json
import tensorflow as tf
from typing import Tuple
from tqdm.auto import tqdm
from plot_keras_history import plot_history
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedShuffleSplit
from epigenomic_dataset import load_epigenomes
from collections import Counter
from sklearn.utils import class_weight


def get_holdout(train: np.ndarray, test: np.ndarray, bed: pd.DataFrame, labels: np.ndarray, genome, batch_size=1024) -> Tuple[Sequence, Sequence]:
    return (
        MixedSequence(
            x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
            y=labels[train],
            batch_size=batch_size
        ),
        MixedSequence(
            x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
            y=labels[test],
            batch_size=batch_size
        )
    )


def precomputed(results, model: str, holdout: int) -> bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
        (df.model == model) &
        (df.holdout == holdout)
    ).any()


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]


def train_model_seq(models, epigenomes, nlabels, region_type, cell_line):
    # Reprod
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)

    splits = 11
    holdouts = StratifiedShuffleSplit(
        n_splits=splits, test_size=0.2, random_state=42)
    genome = Genome("hg19")
    bed = to_bed(epigenomes[region_type])
    labels = nlabels[region_type].values.ravel()

    '''
    if os.path.exists( cell_line + "_" + region_type + "_sequence.json"):
        results = compress_json.local_load( cell_line + "_" + region_type + "_sequence.json")
    else:
    '''
    results = []
    class_w = class_weight.compute_class_weight(
        'balanced', np.unique(labels), labels)
    class_w = dict(enumerate(class_w))
    print("Class weights: " + str(class_w))

    for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        train, test = get_holdout(
            train_index, test_index, bed, labels, genome, 1024)
        print("="*80)
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            if precomputed(results, model.name, i):
                continue
            history = model.fit(
                train,
                steps_per_epoch=train.steps_per_epoch,
                validation_data=test,
                validation_steps=test.steps_per_epoch,
                epochs=1000,
                shuffle=True,
                verbose=False,
                class_weight=class_w,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min",
                                  patience=30, restore_best_weights=True),
                ]
            ).history
            scores = pd.DataFrame(history).iloc[-1].to_dict()
            results.append({
                "model": model.name,
                "run_type": "train",
                "holdout": i,
                **{
                    key: value
                    for key, value in scores.items()
                    if not key.startswith("val_")
                }
            })
            results.append({
                "model": model.name,
                "run_type": "test",
                "holdout": i,
                **{
                    key[4:]: value
                    for key, value in scores.items()
                    if key.startswith("val_")
                }
            })
            compress_json.local_dump(
                results, cell_line + "_" + region_type + "_sequence.json")
            df = pd.DataFrame(results).drop(columns="holdout")
    return df
