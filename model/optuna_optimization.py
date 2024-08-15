import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
import sklearn.datasets
import sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

data = pd.read_csv("finalfeatures.csv")

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

def objective(trial):
    dtrain = lgb.Dataset(X_train, label=y_train)

    # defining the hyperparameter search space
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.007, 0.7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.91),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
    }

    gbm = lgb.train(param, dtrain)

    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)

    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy

# optuna study to maximize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("Value: {}".format(trial.value))
print("Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))   


best_params = trial.params
best_lgbm = lgb.LGBMClassifier(**best_params)
best_lgbm.fit(X_train, y_train)

y_pred = best_lgbm.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("ROC-AUC Score: {:.4f}".format(roc_auc))
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
