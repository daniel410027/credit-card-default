import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_error',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'random_state': seed,
        'device': 'cpu'
    }

    lgbm_opt = LGBMClassifier(**param)
    sc = cross_val_score(lgbm_opt, X_train, y_train, cv=4, scoring='roc_auc').mean()

    return sc

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the best score
print('Best ROC-AUC:', study.best_value)
print('Best hyperparameters:', study.best_params)

param1 = {'n_estimators': 844, 'max_depth': 5, 'learning_rate': 0.011533926188770152, 
          'min_child_weight': 0.7353926016580375, 'min_child_samples': 22, 
          'subsample': 0.7440626280651244, 'subsample_freq': 3
#           , 'colsample_bytree': 0.5193554106905941
         }