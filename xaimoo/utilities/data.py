import pandas as pd 
import numpy as np 

def label_two_bar_truss(file_path: str, delimiter: str = ';'):
    df_all = pd.read_csv(file_path, delimiter=delimiter)

    variable_names = ["x1", "x2", "y"]
    objective_names = ["Volume", "MaxStress"]

    feasible = df_all["Feasible"].values == 1

    df_labeled = df_all[objective_names + variable_names].loc[feasible].copy().reset_index(drop=True)

    """
    Category 1 -- Knee: MaxStress <= 50000 & Volume <= 0.02
    Category 2 -- LeftExtreme: MaxStress >= 70000 & Volume <= 0.015
    Category 3 -- Volume >= 0.055
    Category 4 -- MaxStress >= 50000 & Volume >= 0.04
    """

    # Category 1
    c1_mask = np.logical_and(df_labeled["MaxStress"] <= 50000, df_labeled["Volume"] <= 0.02)
    # Category 2
    c2_mask = np.logical_and(df_labeled["MaxStress"] >= 70000, df_labeled["Volume"] <= 0.015)
    # Category 3
    c3_mask = df_labeled["Volume"] >= 0.055
    # Category 4
    c4_mask = np.logical_and(df_labeled["MaxStress"] >= 50000, df_labeled["Volume"] >= 0.04)

    df_labeled.loc[c1_mask, "category"] = 1
    df_labeled.loc[c2_mask, "category"] = 2
    df_labeled.loc[c3_mask, "category"] = 3
    df_labeled.loc[c4_mask, "category"] = 4

    df_labeled_clean = df_labeled[~df_labeled["category"].isnull()]

    return df_labeled_clean


def label_vehicle_crash(file_path: str, delimiter: str = ';'):
    df_all = pd.read_csv(file_path, delimiter=delimiter)


    variable_names = ["x1", "x2", "x3", "x4", "x5"]
    objective_names = ["Weight", "Acceleration", "Intrusion"]

    feasible = df_all["Feasible"].values == 1

    df_labeled = df_all[objective_names + variable_names].loc[feasible].copy().reset_index(drop=True)

    """ 
    Category 1 -- Knee: Weight <= 1685 & Acceleration <= 8.5 & Intrusion <= 0.10
    Category 2 -- F1Extreme: Weight >= 1695
    Category 3 -- F2Extreme: Acceleration >= 10.5 & Weight <= 1685
    Category 4 -- F3Extreme: Intrusion >= 0.20
    Category 5 -- Dominated: Acceleration >= 9.0 & Intrusion >= 0.15
    """

    # Category 1
    c1_mask = np.logical_and(df_labeled["Weight"] <= 1685, df_labeled["Acceleration"] <= 8.5, df_labeled["Intrusion"] <= 0.10)
    # Category 2
    c2_mask = df_labeled["Weight"] >= 1695
    # Category 3
    c3_mask = np.logical_and(df_labeled["Acceleration"] >= 10.5, df_labeled["Weight"] <= 1685)
    # Category 4
    c4_mask = df_labeled["Intrusion"] >= 0.20
    # Category 5
    c5_mask = np.logical_and(df_labeled["Acceleration"] >= 9.0, df_labeled["Intrusion"] >= 0.15)


    df_labeled.loc[c1_mask, "category"] = 1
    df_labeled.loc[c2_mask, "category"] = 2
    df_labeled.loc[c3_mask, "category"] = 3
    df_labeled.loc[c4_mask, "category"] = 4
    df_labeled.loc[c5_mask, "category"] = 5

    df_labeled_clean = df_labeled[~df_labeled["category"].isnull()]

    return df_labeled_clean, variable_names, objective_names


def label_welded_beam(file_path: str, delimiter: str = ';'):
    df_all = pd.read_csv(file_path, delimiter=delimiter)

    variable_names = ["x1", "x2", "x3", "x4"]
    objective_names = ["Cost", "Deflection"]

    feasible = df_all["Feasible"].values == 1

    df_labeled = df_all[objective_names + variable_names].loc[feasible].copy().reset_index(drop=True)

    """
    Knee: Cost <= 15 & Deflection <= 0.004
    LeftExtreme: Cost <= 10 & Deflection >= 0.005
    RightExtreme: Cost >= 150 & Deflection <= 0.005
    Dominated: Cost >= 80 & Deflection >= 0.020
    """

    # Category 1
    c1_mask = np.logical_and(df_labeled["Cost"] <= 15.0, df_labeled["Deflection"] <= 0.004)
    # Category 2
    c2_mask = np.logical_and(df_labeled["Cost"] <= 10.0, df_labeled["Deflection"] >= 0.005)
    # Category 3
    c3_mask = np.logical_and(df_labeled["Cost"] >= 150.0, df_labeled["Deflection"] <= 0.005)
    # Category 4
    c4_mask = np.logical_and(df_labeled["Cost"] >= 80.0, df_labeled["Deflection"] >= 0.020)

    df_labeled.loc[c1_mask, "category"] = 1
    df_labeled.loc[c2_mask, "category"] = 2
    df_labeled.loc[c3_mask, "category"] = 3
    df_labeled.loc[c4_mask, "category"] = 4

    df_labeled_clean = df_labeled[~df_labeled["category"].isnull()]

    return df_labeled_clean


if __name__ == "__main__":
    #res = label_two_bar_truss("../data/TwoBarTruss.csv")
    #res = label_vehicle_crash("../data/VehicleCrash.csv")
    res = label_welded_beam("../data/WeldedBeam.csv")
    res = print(res)
