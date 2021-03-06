import pandas as pd
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from multiprocessing import cpu_count


def get_features_filter(X: pd.DataFrame, y: pd.DataFrame, name: str, cicli:int) -> BorutaPy:
    boruta_selector = BorutaPy(
        RandomForestClassifier(
            n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
        n_estimators='auto',
        verbose=2,
        alpha=0.05,
        max_iter=cicli,
        random_state=42
    )
    boruta_selector.fit(X.values, y.values.ravel())
    return boruta_selector


def filter_epigenome(cell_line, epigenomes, labels, cicli):
    filtered_epigenomes = {
        region: get_features_filter(
            X=x,
            y=labels[region],
            name=f"{cell_line}/{region}",
            cicli=cicli
        ).transform(x.values)
        for region, x in tqdm(
            epigenomes.items(),
            desc="Running Baruta Feature estimation"
        )
    }

    return filtered_epigenomes
