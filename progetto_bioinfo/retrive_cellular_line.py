import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from epigenomic_dataset import load_epigenomes
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


def robust_zscoring(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame without constant features."""
    return df.loc[:, (df != df.iloc[0]).any()]


def knn_imputer(df: pd.DataFrame, neighbours: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        KNNImputer(n_neighbors=neighbours).fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def retrive_cell_line(line, win_size):
    # Reprod
    np.random.seed(42)

    cell_line = line
    window_size = win_size

    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1)
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)

    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }
    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }

    # Il risultato dovrebbe essere >> 1
    for region, x in epigenomes.items():
        print(
            f"Il rate tra features e samples per i dati {region} é di: {x.shape[0]/x.shape[1]}"
        )
        print("="*80)

    # Presenza di Nan
    for region, x in epigenomes.items():
        print("\n".join((
            f"Controllo Nan in {region} data:",
            f"Sono presenti {x.isna().values.sum()} NaN su {x.values.size} valori."
        )))
        print("="*80)

    # Knn inputation
    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)

    # Controllo class balance
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for axis, (region, y) in zip(axes.ravel(), labels.items()):
        y.hist(ax=axis, bins=3)
        axis.set_title(f"Classes count in {region}")
    fig.savefig("./imgs/" + cell_line + f"/class_balance")

    # Se presenti feature costanti vanno rimosse
    for region, x in epigenomes.items():
        result = drop_constant_features(x)
        if x.shape[1] != result.shape[1]:
            print(f"In {region} le feature costanti sono state rimosse!")
            epigenomes[region] = result
        else:
            print(f"Nessuna feature costante in {region}!")

    # Normalizziamo con z-scoring
    epigenomes = {
        region: robust_zscoring(x)
        for region, x in epigenomes.items()
    }

    return epigenomes, labels
