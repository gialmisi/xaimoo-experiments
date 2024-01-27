import numpy as np
import pandas as pd
import re
from imodels import SkopeRulesClassifier, RuleFitClassifier, BayesianRuleSetClassifier


def train_skope_rules(
    labeled_df: pd.DataFrame,
    variable_names: list[str],
    target_cetegory: int,
    target_column: str = "category",
    classifier_kwargs: dict | None = None,
) -> SkopeRulesClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(
        lambda x: 1 if int(x) == target_cetegory else 0
    )
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
        # args[0] == precision, args[1] == recall
        f1_score = 2 * (rule.args[0] * rule.args[1]) / (rule.args[0] + rule.args[1])
        rules[f"{rule}"] = (rule.args[0], rule.args[1], f1_score)

    # sort according to f1_score
    rules = dict(sorted(rules.items(), key=lambda item: item[1][2], reverse=True))

    print("Rule --> (Accuracy, Recall, F1-score)")
    for i, rule in enumerate(rules):
        print(f"{i}: {rule} --> {tuple(f'{e:.3f}' for e in rules[rule])}")

    return rules


def train_rulefit_rules(
    labeled_df: pd.DataFrame,
    variable_names: list[str],
    target_cetegory: int,
    target_column: str = "category",
    classifier_kwargs: dict | None = None,
) -> SkopeRulesClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(
        lambda x: 1 if int(x) == target_cetegory else 0
    )
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
    for rule, rule_type, support, importance in zip(
        classifier_rules["rule"],
        classifier_rules["type"],
        classifier_rules["support"],
        classifier_rules["importance"],
    ):
        if rule_type == "rule":
            rules[f"{rule}"] = (support, importance)

    # sort according to support
    rules = dict(sorted(rules.items(), key=lambda item: item[1][0], reverse=True))

    print("Rule --> (Support, Importance)")
    for i, rule in enumerate(rules):
        print(f"{i}: {rule} --> {rules[rule]}")

    return rules


def train_bayesian_rules(
    labeled_df: pd.DataFrame,
    variable_names: list[str],
    target_cetegory: int,
    target_column: str = "category",
    classifier_kwargs: dict | None = None,
) -> BayesianRuleSetClassifier:
    df_copy = labeled_df.copy()

    df_copy["category"] = df_copy["category"].apply(
        lambda x: 1 if int(x) == target_cetegory else 0
    )
    x_train = df_copy[variable_names]
    y_target = df_copy["category"]

    if classifier_kwargs is not None:
        classifier = BayesianRuleSetClassifier(**classifier_kwargs)
    else:
        classifier = BayesianRuleSetClassifier()

    classifier.fit(X=x_train, y=y_target)

    return classifier


def rule_to_condition_list(rule: str) -> list:
    conditions = rule.split(" and ")
    cons = []

    for condition in conditions:
        # Variable names assumed to be either of type x or y optionally followed by numbers, i.e., "x1" or y"
        # throw away the index in '_', already included in 'variable'
        variable, _, operator, value = re.match(
            r"((x|y)\d*)\s*(<=|<|>=|>|==|!=)\s*([\d.]+)", condition
        ).groups()

        cons.append((variable, operator, value))

    return cons


def rule_to_conditions(rule: str):
    cons = rule_to_condition_list(rule)

    def _checker(values: pd.DataFrame, cons: list = cons) -> bool:
        is_ok = []
        for var_name, op, val in cons:
            val = float(val)
            match op:
                case ">":
                    tmp = values[var_name] > val
                case ">=":
                    tmp = values[var_name] >= val
                case "==":
                    tmp = values[var_name] == val
                case "!=":
                    tmp = values[var_name] != val
                case "<":
                    tmp = values[var_name] < val
                case "<=":
                    tmp = values[var_name] <= val
                case _:
                    raise ValueError(f"Operator {op} not supported.")

            is_ok.append(tmp)

        return all(is_ok)

    return _checker


def combine_rule_conditions(rules: list[str]):
    checkers = []

    for rule in rules:
        checker = rule_to_conditions(rule)
        checkers.append(checker)

    def _callable(values: pd.DataFrame, checkers=checkers):
        for _checker in checkers:
            res = _checker(values)
            if not res:
                return False

        # if all checkers pass, return True
        return True

    return _callable


def index_rules(rules: dict) -> dict:
    indexed_rules = {}
    for i, rule in enumerate(rules.items()):
        indexed_rules[i] = rule

    return indexed_rules


def simplify_rules(rules: list[str]) -> str:
    var_rules = {}

    for rule in rules:
        cons = rule_to_condition_list(rule)

        for var, op, val in cons:
            if not var in var_rules:
                # no rule for var
                var_rules[var] = {op: val}
            else:
                if not op in var_rules[var]:
                    # no rule for var with op
                    var_rules[var][op] = val
                else:
                    # rule exists for var with op, check for redundancy
                    match op:
                        case ">":
                            if val > var_rules[var][op]:
                                var_rules[var][op] = val
                        case ">=":
                            if val >= var_rules[var][op]:
                                var_rules[var][op] = val
                        case "==":
                            pass
                        case "!=":
                            pass
                        case "<":
                            if val < var_rules[var][op]:
                                var_rules[var][op] = val
                        case "<=":
                            if val <= var_rules[var][op]:
                                var_rules[var][op] = val
                        case _:
                            raise ValueError(f"Operator {op} not supported.")

    simple_rules = []

    for var_name in var_rules:
        simple_rule = []
        if ">" in var_rules[var_name]:
            simple_rule.append(f"{var_name} > {var_rules[var_name]['>']}")
        if ">=" in var_rules[var_name]:
            simple_rule.append(f"{var_name} >= {var_rules[var_name]['>=']}")
        if "<" in var_rules[var_name]:
            simple_rule.append(f"{var_name} < {var_rules[var_name]['<']}")
        if "<=" in var_rules[var_name]:
            simple_rule.append(f"{var_name} <= {var_rules[var_name]['<=']}")

        simple_rules.append(" & ".join(simple_rule))

    return " AND ".join(simple_rules)


if __name__ == "__main__":
    from xaimoo.utilities.data import label_vehicle_crash

    """
    df_crash, var_names, obj_names = label_vehicle_crash("./data/VehicleCrash.csv")
    kw_args = {"max_depth": range(1, len(var_names)+1), "precision_min": 0.7, "recall_min": 0.7}

    
    # classifier = train_skope_rules(df_crash, var_names, 1, classifier_kwargs=kw_args)

    # res = explain_skope_rules(classifier)

    rule = "x3 <= 2.63713 and x4 <= 2.55"
    rule2 = "x1 < 2"

    test_frame = pd.DataFrame({"x1": [1,2,3], "x2": [1,2,3], "x3": [1.5, 2.5, 3.0], "x4": [1.5, 2.5, 2.6]})

    funs = rule_to_conditions(rule)
    funs2 = rule_to_conditions(rule2)
    rules = [rule, rule2]
    funs_comb = combine_rule_conditions(rules)
    # test_frame["rule"] = test_frame.apply(lambda row: funs(row) & funs2(row), axis=1)
    test_frame["rule"] = test_frame.apply(lambda row: funs_comb(row), axis=1)
    print(" AND ".join(rules))
    print(test_frame)


    kw_args = {"max_depth": range(1, len(var_names)+1), "precision_min": 0.7, "recall_min": 0.5}

    classifier = train_skope_rules(df_crash, var_names, 1, classifier_kwargs=kw_args)

    res = explain_skope_rules(classifier)
    
    print(res)
    print(index_rules(res))
    """

    rules = ["x1 < 0.8 and x3 > 0.2", "x1 < 0.5 and x3 > 0.1", "x2 < 0.5", "x1 > 0.2"]

    print(simplify_rules(rules))
