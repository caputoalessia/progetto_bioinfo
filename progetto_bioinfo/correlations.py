import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.stats import entropy, spearmanr, pearsonr
from minepy import MINE
from sklearn.metrics.pairwise import euclidean_distances


def get_top_most_different_tuples(dist, n: int):
    return list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:n]


def get_top_most_different(dist, n: int):
    return np.argsort(-np.mean(dist, axis=1).flatten())[:n]


def get_correlations(cell_line, epigenomes, labels):
    # Reprod
    np.random.seed(42)

    # Correlazione
    p_value_threshold = 0.01
    correlation_threshold = 0.1

    uncorrelated = {
        region: set()
        for region in epigenomes
    }

    # Pearson
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = pearsonr(
                x[column].values.ravel(), labels[region].values.ravel())
            #print(region, column, correlation)
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)

    # Speamer
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(
                x[column].values.ravel(), labels[region].values.ravel())
            #print(region, column, correlation)
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    # Correlazione non lineare
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True, leave=False):
            mine = MINE()
            mine.compute_score(x[column].values.ravel(),
                               labels[region].values.ravel())
            score = mine.mic()
            if score < correlation_threshold:
                print(region, column, score)
            else:
                uncorrelated[region].remove(column)
    # Rimozione feature senza correlazione
    for region, x in epigenomes.items():
        epigenomes[region] = x.drop(columns=[
            col
            for col in uncorrelated[region]
            if col in x.columns
        ])
    # Correlaizione
    p_value_threshold = 0.01
    correlation_threshold = 0.95
    extremely_correlated = {
        region: set()
        for region in epigenomes
    }

    scores = {
        region: []
        for region in epigenomes
    }

    for region, x in epigenomes.items():
        for i, column in tqdm(
                enumerate(x.columns),
                total=len(x.columns), desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            for feature in x.columns[i+1:]:
                correlation, p_value = pearsonr(
                    x[column].values.ravel(), x[feature].values.ravel())
                correlation = np.abs(correlation)
                scores[region].append((correlation, column, feature))
                if p_value < p_value_threshold and correlation > correlation_threshold:
                    print(region, column, feature, correlation)
                    if entropy(x[column]) > entropy(x[feature]):
                        extremely_correlated[region].add(feature)
                    else:
                        extremely_correlated[region].add(column)

    scores = {
        region: sorted(score, key=lambda x: np.abs(x[0]), reverse=True)
        for region, score in scores.items()
    }

    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][:3]))
        columns = list(set(firsts+seconds))
        print(f"Most correlated features from {region} epigenomes")
        sns_plot = sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        sns_plot.savefig("./imgs/" + cell_line +
                         f"/Most_correlated_{region}.png")

    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][-3:]))
        columns = list(set(firsts+seconds))
        print(f"Least correlated features from {region} epigenomes")
        sns_plot = sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        sns_plot.savefig("./imgs/" + cell_line +
                         f"/Least_correlated_{region}.png")

    # Piú differenti
    top_number = 5

    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        most_distance_columns_indices = get_top_most_different(
            dist, top_number)
        columns = x.columns[most_distance_columns_indices]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        print(f"Top {top_number} different features from {region}.")
        for column, axis in zip(columns, axes.flatten()):
            head, tail = x[column].quantile([0.05, 0.95]).values.ravel()

            mask = ((x[column] < tail) & (x[column] > head)).values

            cleared_x = x[column][mask]
            cleared_y = labels[region].values.ravel()[mask]

            cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
            cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

            axis.set_title(column)
        fig.tight_layout()
        fig.savefig("./imgs/" + cell_line +
                    f"/Top_{top_number}_different_features_{region}.png")

    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        dist = np.triu(dist)
        tuples = get_top_most_different_tuples(dist, top_number)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        print(f"Top {top_number} different tuples of features from {region}.")
        for (i, j), axis in zip(tuples, axes.flatten()):
            column_i = x.columns[i]
            column_j = x.columns[j]
            for column in (column_i, column_j):
                head, tail = x[column].quantile([0.05, 0.95]).values.ravel()
                mask = ((x[column] < tail) & (x[column] > head)).values
                x[column][mask].hist(ax=axis, bins=20, alpha=0.5)
            axis.set_title(f"{column_i} and {column_j}")
        fig.tight_layout()
        fig.savefig("./imgs/" + cell_line +
                    f"/Top_{top_number}_different_tuples_{region}.png")

    return epigenomes
