import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import itertools

def plot_rule_explanations(df: pd.DataFrame, objective_names: list[str], rule_check_column: str, true_check_colum: str, truth_value: int, title: str = ""):
    rule_group = df[df[rule_check_column] == True]
    no_rule_group = df[df[rule_check_column] == False]
    true_group = df[df[true_check_colum] == truth_value]
    false_pos_group = df[(df[true_check_colum] != truth_value) & (df[rule_check_column] == True)]
    false_neg_group = df[(df[true_check_colum] == truth_value) & (df[rule_check_column] == False)]

    base_size = 5
    if len(objective_names) > 2:
        fig, axs = plt.subplots(1, len(objective_names), figsize=(len(objective_names)*base_size, base_size))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(1*base_size, base_size))
        axs = [axs]

    objective_pairs = itertools.combinations(objective_names, 2)

    for i, pair in enumerate(objective_pairs):
        axs[i].scatter(no_rule_group[pair[0]], no_rule_group[pair[1]], c="red", s=1, alpha=0.5, label="False")
        axs[i].scatter(false_neg_group[pair[0]], false_neg_group[pair[1]], c="blue", s=5, marker="v", alpha=0.3, label="False neg")
        axs[i].scatter(false_pos_group[pair[0]], false_pos_group[pair[1]], c="orange", s=5, marker="^", alpha=0.3, label="False pos")
        axs[i].scatter(rule_group[pair[0]], rule_group[pair[1]], c="green", s=1, alpha=0.5, label="True")

        points = true_group[[pair[0], pair[1]]].to_numpy()
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            axs[i].plot(points[simplex, 0], points[simplex, 1], "-", c="black")

        axs[i].set_xlabel(pair[0])
        axs[i].set_ylabel(pair[1])
        axs[i].legend()

    plt.suptitle(title)

    plt.show()