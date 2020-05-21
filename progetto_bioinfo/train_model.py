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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedShuffleSplit
from epigenomic_dataset import load_epigenomes

def get_holdout(train:np.ndarray, test:np.ndarray, bed:pd.DataFrame, labels:np.ndarray, genome, batch_size=1024)->Tuple[Sequence, Sequence]:
    return (
        MixedSequence(
            x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
            y=labels[train],
            batch_size=batch_size
        ),
        MixedSequence(
            x= BedSequence(genome, bed.iloc[test], batch_size=batch_size),
            y=labels[test],
            batch_size=batch_size
        )
    )

def precomputed(results, model:str, holdout:int)->bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
        (df.model == model) &
        (df.holdout == holdout)
    ).any()

def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]

def train_model(model, epigenomes, nlabels):
    splits = 2
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)
    genome = Genome("hg19")
    bed = epigenomes["promoters"].reset_index()[epigenomes["promoters"].index.names]
    print(bed.shape)
    labels = nlabels["promoters"].values.ravel()


    if os.path.exists("sequence.json"):
        results = compress_json.local_load("sequence.json")
    else:
        results = []

    for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        train, test = get_holdout(train_index, test_index, bed, labels, genome)
        if precomputed(results, model.name, i):
            continue
        history = model.fit(
            train,
            steps_per_epoch=train.steps_per_epoch,
            validation_data=test,
            validation_steps=test.steps_per_epoch,
            epochs=1000,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        ).history
        scores = pd.DataFrame(history).iloc[-1].to_dict()
        results.append({
            "model":model.name,
            "run_type":"train",
            "holdout":i,
            **{
                key:value
                for key, value in scores.items()
                if not key.startswith("val_")
            }
        })
        results.append({
            "model":model.name,
            "run_type":"test",
            "holdout":i,
            **{
                key[4:]:value
                for key, value in scores.items()
                if key.startswith("val_")
            }
        })
        compress_json.local_dump(results, "sequence.json")
        df = pd.DataFrame(results).drop(columns="holdout")
    return df
    
