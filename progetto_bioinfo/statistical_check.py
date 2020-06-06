from scipy.stats import wilcoxon
import compress_json
import pandas as pd
import numpy as np


def get_df(cell_line, data_type, region_type):
    results = compress_json.local_load(
        cell_line + "_" + region_type + "_" + data_type + ".json")
    df = pd.DataFrame(results).drop(columns="holdout")
    return df[(df.run_type == "test")]


def wilcoxon_test(modela_scores, modelb_scores):
    alpha = 0.01

    for metric in modela_scores.columns[-4:]:
        print(metric)
        a,  b = modela_scores[metric], modelb_scores[metric]
        stats, p_value = wilcoxon(a, b)
        if p_value > alpha:
            print(p_value, "The two models performance are statistically identical.")
        else:
            print(p_value, "The two models performance are different")
            if a.mean() > b.mean():
                print("The first model is better")
            else:
                print("The second model is better")


def do_test(cell_line, data_type, region_type, model_list):

    models = get_df(cell_line, data_type, region_type)
    modela_scores = models[models.model == model_list[0].name]
    modelb_scores = models[models.model == model_list[1].name]
    wilcoxon_test(modela_scores, modelb_scores)
