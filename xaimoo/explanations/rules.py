import numpy as np 
import pandas as pd
import re
from imodels import SkopeRulesClassifier, RuleFitClassifier, BayesianRuleSetClassifier

def train_skope_rules(labeled_df: pd.DataFrame, variable_names: list[str], target_cetegory: int, target_column: str = "category", classifier_kwargs: dict | None = None) -> SkopeRulesClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(lambda x: 1 if int(x) == target_cetegory else 0)
    x_train = df_copy[variable_names]
    y_target = df_copy["category"]

    if classifier_kwargs is not None:
        classifier = SkopeRulesClassifier(**classifier_kwargs)
    else:
        classifier = SkopeRulesClassifier()

    classifier.fit(X=x_train, y=y_target)

    return classifier

def explain_skope_rules(classifier: SkopeRulesClassifier) -> dict:
    rules = {}
    for rule in classifier.rules_:
        rules[f"{rule}"] = (rule.args[0], rule.args[1])

    # sort according to accuracy
    rules = dict(sorted(rules.items(), key=lambda item: item[1][0], reverse=True))
    
    print("Rule --> (Accuracy, Recall)")
    for rule in rules:
        print(f"{rule} --> {rules[rule]}")

    return rules

def train_rulefit_rules(labeled_df: pd.DataFrame, variable_names: list[str], target_cetegory: int, target_column: str = "category", classifier_kwargs: dict | None = None) -> SkopeRulesClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(lambda x: 1 if int(x) == target_cetegory else 0)
    x_train = df_copy[variable_names]
    y_target = df_copy["category"]

    if classifier_kwargs is not None:
        classifier = RuleFitClassifier(**classifier_kwargs)
    else:
        classifier = RuleFitClassifier()

    classifier.fit(X=x_train, y=y_target)

    return classifier


def explain_rulefit_rules(classifier: RuleFitClassifier) -> dict:
    rules = {}
    classifier_rules = classifier._get_rules()
    for rule, rule_type, support, importance in zip(classifier_rules["rule"], classifier_rules["type"], classifier_rules["support"], classifier_rules["importance"]):
        if rule_type == "rule":
            rules[f"{rule}"] = (support, importance)

    # sort according to support
    rules = dict(sorted(rules.items(), key=lambda item: item[1][0], reverse=True))

    print("Rule --> (Support, Importance)")
    for rule in rules:
        print(f"{rule} --> {rules[rule]}")

    return rules

def train_bayesian_rules(labeled_df: pd.DataFrame, variable_names: list[str], target_cetegory: int, target_column: str = "category", classifier_kwargs: dict | None = None) -> BayesianRuleSetClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(lambda x: 1 if int(x) == target_cetegory else 0)
    x_train = df_copy[variable_names]
    y_target = df_copy["category"]

    if classifier_kwargs is not None:
        classifier = BayesianRuleSetClassifier(**classifier_kwargs)
    else:
        classifier = BayesianRuleSetClassifier()

    classifier.fit(X=x_train, y=y_target)

    return classifier


def rule_to_conditions(rule: str):
    conditions = rule.split(" and ")
    cons = []  

    for condition in conditions:
        variable, operator, value = re.match(r'(x\d+)\s*(<=|<|>=|>|==|!=)\s*([\d.]+)', condition).groups()
        cons.append((variable, operator, value))


    def _checker(values: pd.DataFrame, cons: list = cons) -> bool:
        for var_name, op, val in cons:
            print(var_name, op, val)
            pass
        pass

    return _checker

if __name__ == "__main__":
    from xaimoo.utilities.data import label_vehicle_crash
    df_crash, var_names, obj_names = label_vehicle_crash("../../data/VehicleCrash.csv")
    """
    kw_args = {"max_depth": range(1, len(var_names)+1), "precision_min": 0.7, "recall_min": 0.7}

    classifier = train_skope_rules(df_crash, var_names, 1, classifier_kwargs=kw_args)

    res = explain_skope_rules(classifier)

    classifier = train_rulefit_rules(df_crash, var_names, 2)
    explain_rulefit_rules(classifier)
    """

    rule = "x3 <= 2.63713 and x4 <= 1.70858"

    test_frame = pd.DataFrame({"x1": [1,2,3], "x2": [1,2,3], "x3": [1.5, 2.5, 3.0], "x4": [1.5, 2.5, 2.6]})

    fun = rule_to_conditions(rule)
    fun(test_frame)
    