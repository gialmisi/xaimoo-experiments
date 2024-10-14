import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import itertools


def plot_rule_explanations(
    df: pd.DataFrame,
    objective_names: list[str],
    rule_check_column: str,
    true_check_colum: str,
    truth_value: int,
    title: str = "",
):
    """
    Plots rule explanations by visualizing the classification of data based on a given rule
    and the true values. The function creates scatter plots for pairs of objectives and overlays
    convex hulls to highlight the boundaries of true and rule-satisfying groups.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        objective_names (list[str]): A list of column names (features) to be used for the scatter plots.
        rule_check_column (str): The column name that indicates whether the rule is satisfied (boolean values).
        true_check_colum (str): The column name that represents the true labels or expected outcomes.
        truth_value (int): The value in `true_check_colum` that is considered the "true" or positive class.
        title (str, optional): A title for the plot. Defaults to an empty string.

    Prints:
        - True positives: Count of correct classifications where the rule and true label match.
        - True negatives: Count of correct classifications where the rule is not satisfied and the true label differs.
        - False positives: Count of incorrect classifications where the rule is satisfied but the true label does not match.
        - False negatives: Count of incorrect classifications where the rule is not satisfied but the true label matches.

    Plots:
        - A set of scatter plots for each pair of objectives in `objective_names`.
        - Data points where the rule is satisfied (True) are marked in green.
        - Data points where the rule is not satisfied (False) are marked in blue.
        - Convex hulls are drawn around the points in the true group (solid blue) and rule group (dashed green).

    Example:
        ```python
        import pandas as pd

        data = {
            "objective_1": [1.0, 2.5, 3.0, 4.2, 5.5, 6.0, 7.8, 8.5],
            "objective_2": [1.5, 3.0, 2.7, 4.5, 5.2, 7.0, 8.2, 9.5],
            "rule_check": [True, False, True, False, True, True, False, True],
            "true_label": [1, 0, 1, 0, 1, 1, 0, 1],
        }

        df = pd.DataFrame(data)

        plot_rule_explanations(df, ['objective_1', 'objective_2'], 'rule_check', 'true_label', truth_value=1)
        ```
    """

    rule_group = df[df[rule_check_column] == True]
    no_rule_group = df[df[rule_check_column] == False]
    true_group = df[df[true_check_colum] == truth_value]

    print(
        f"True positives  = {sum((df[true_check_colum] == truth_value) & (df[rule_check_column] == True))} / {sum(df[true_check_colum] == truth_value)}"
    )
    print(
        f"True negatives  = {sum((df[true_check_colum] != truth_value) & (df[rule_check_column] == False))} / {sum(df[true_check_colum] != truth_value)}"
    )
    print(
        f"False positives = {sum((df[true_check_colum] != truth_value) & (df[rule_check_column] == True))} / {sum(df[rule_check_column] == True)}"
    )
    print(
        f"False negatives = {sum((df[true_check_colum] == truth_value) & (df[rule_check_column] == False))} / {sum(df[rule_check_column] == False)}"
    )

    base_size = 5
    if len(objective_names) > 2:
        fig, axs = plt.subplots(
            1,
            len(objective_names),
            figsize=(len(objective_names) * base_size, base_size),
        )
    else:
        fig, axs = plt.subplots(1, 1, figsize=(1 * base_size, base_size))
        axs = [axs]

    objective_pairs = itertools.combinations(objective_names, 2)

    for i, pair in enumerate(objective_pairs):
        axs[i].scatter(
            no_rule_group[pair[0]],
            no_rule_group[pair[1]],
            c="#a6cee3",
            s=1,
            alpha=1.0,
            label="False",
        )
        axs[i].scatter(
            rule_group[pair[0]],
            rule_group[pair[1]],
            c="#b2df8a",
            s=1,
            alpha=1.0,
            label="True",
        )

        points = true_group[[pair[0], pair[1]]].to_numpy()
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            axs[i].plot(points[simplex, 0], points[simplex, 1], "-", c="#1f78b4")

        rule_points = rule_group[[pair[0], pair[1]]].to_numpy()
        rule_hull = ConvexHull(rule_points)
        for simplex in rule_hull.simplices:
            axs[i].plot(
                rule_points[simplex, 0], rule_points[simplex, 1], "--", c="#b2df8a"
            )

        axs[i].set_xlabel(pair[0])
        axs[i].set_ylabel(pair[1])
        axs[i].legend()

    plt.suptitle(title)


if __name__ == "__main__":
    data = {
        "objective_1": [1.0, 2.5, 3.0, 4.2, 5.5, 6.0, 7.8, 8.5],
        "objective_2": [1.5, 3.0, 2.7, 4.5, 5.2, 7.0, 8.2, 9.5],
        "rule_check": [True, False, True, False, True, True, False, True],
        "true_label": [1, 0, 1, 0, 1, 1, 0, 1],
    }

    df = pd.DataFrame(data)

    plot_rule_explanations(
        df, ["objective_1", "objective_2"], "rule_check", "true_label", truth_value=1
    )

    plt.show()
