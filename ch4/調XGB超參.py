import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

seed = np.random.seed(6)
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
    # sc = cross_val_score(lgbm_opt, X_train, y_train, cv=4, scoring='roc_auc').mean()
    sc_f1 = cross_val_score(lgbm_opt,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
    return sc_f1
def rfobjective(trial):
    # Number of trees in random forest
    n_estimators = trial.suggest_int(name="n_estimators", low=100, high=500, step=100)

    # Number of features to consider at every split
    max_features = trial.suggest_categorical(name="max_features", choices=['auto', 'sqrt']) 

    # Maximum number of levels in tree
    max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

    # Minimum number of samples required to split a node
    min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=10, step=2)

    # Minimum number of samples required at each leaf node
    min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
    
    params = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }
    model = RandomForestClassifier(random_state=seed, **params)
    
    sc_f1 = cross_val_score(rfc,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
    return sc_f1

def xgbobjective(trial):
    """
    A function to train a model using different hyperparamerters combinations provided by Optuna.
    """
    seed = np.random.seed(7)
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, 100),
        'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0)
    }

    xgb_opt = XGBClassifier(random_state=seed, **params)
    sc_f1 = cross_val_score(xgb_opt,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
    return f1
# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(xgbobjective, n_trials=100)

# Print the best hyperparameters and the best score
print('Best f1:', study.best_value)
print('Best hyperparameters:', study.best_params)













param1 = {'n_estimators': 844, 'max_depth': 5, 'learning_rate': 0.011533926188770152, 
          'min_child_weight': 0.7353926016580375, 'min_child_samples': 22, 
          'subsample': 0.7440626280651244, 'subsample_freq': 3
#           , 'colsample_bytree': 0.5193554106905941
         }