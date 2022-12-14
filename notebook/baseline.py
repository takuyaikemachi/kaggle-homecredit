#%%
from sys import displayhook
import numpy as np
import pandas as pd
import re
import pickle
import gc
import pandas_profiling as pdp

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

#%%
application_train = pd.read_csv("../input/application_train.csv")
print(application_train.shape)
display(application_train.head())

#%%
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

#%%
application_train = reduce_mem_usage(application_train)

#%%
application_train.info()
application_train.describe(exclude='number').T
application_train.isnull().sum()

#%% dataset
x_train = application_train.drop(columns=["TARGET", "SK_ID_CURR"])
y_train = application_train["TARGET"]
id_train = application_train["SK_ID_CURR"]

for col in x_train.columns:
    if x_train[col].dtype == "O":
        x_train[col] = x_train[col].astype("category")

#%% design validation
print("mean: {:.4f}".format(y_train.mean()))
display(y_train.value_counts())

#%%
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))
print ('index(train): ', cv[0][0])
print ('index(valid): ', cv[0][1])

#%%
def train_lgb(input_x, input_y, input_id, params, list_nfold=[0,1,2,3,4], n_splits=5):
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))
    for nfold in list_nfold:
        print("-"*20, nfold, "-"*20)

        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        print (idx_tr)
        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y.loc[idx_tr], input_id.loc[idx_tr]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y.loc[idx_va], input_id.loc[idx_va]
        print (x_tr.shape, x_va.shape)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)],
            early_stopping_rounds=100,
            verbose=100
        )
        fname_lgb = "../model/model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)
        
        y_tr_pred = model.predict_proba(x_tr)[:, 1]
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print("(auc) tr: {:.4f}, va: {:.4f}".format(metric_tr, metric_va))

        train_oof[idx_va] = y_va_pred

        _imp = pd.DataFrame({
            "col": input_x.columns,
            "imp": model.feature_importances_,
            "nfold": nfold
        })
        imp = pd.concat([imp, _imp])

    print("-"*20, "result", "-"*20)
    metrics = np.array(metrics)
    print (metrics)
    print("[cv] tr: {:.4f}+-{:.4f}, va: {:.4f}+-{:.4f}".format(
        metrics[:, 1].mean(), metrics[:, 1].std(),
        metrics[:, 2].mean(), metrics[:, 2].std()
    ))

    print("(oof) {:.4f}".format(
        roc_auc_score(input_y, train_oof)
    ))

    train_oof = pd.concat([
        input_id, pd.DataFrame({"pred": train_oof})
    ], axis=1)

    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    return train_oof, imp, metrics

#%%
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 32,
    'n_estimators': 100000,
    'random_state': 123,
    'importance_type': 'gain'
}

train_oof, imp, metrics = train_lgb(
    x_train, y_train, id_train, params, list_nfold=[0,1,2,3,4], n_splits=5
)

#%%
def predict_lgb(input_x, input_id, list_nfold=[0,1,2,3,4]):
    pred = np.zeros((len(input_x), len(list_nfold)))
    for nfold in list_nfold:
        print('-'*20, nfold, '-'*20)
        fname_lgb = "../model/model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

    pred = pd.concat([
        input_id, pd.DataFrame({"pred": pred.mean(axis=1)})
    ], axis=1)

    print('Done.')

    return pred

#%%
application_test = pd.read_csv('../input/application_test.csv')
application_test = reduce_mem_usage(application_test)

x_test = application_test.drop(columns=["SK_ID_CURR"])
id_test = application_test[["SK_ID_CURR"]]

for col in x_test.columns:
    if x_test[col].dtype == 'O':
        x_test[col] = x_test[col].astype('category')

test_pred = predict_lgb(x_test, id_test, list_nfold=[0,1,2,3,4])

df_submit = test_pred.rename(columns={'pred': 'TARGET'})
print(df_submit.shape)
display(df_submit.head())

df_submit.to_csv('../output/submission_baseline.csv', index=None)

#%%
display(application_train["DAYS_EMPLOYED"].value_counts())
print ('% of positive values: {:.0%}'.format((application_train["DAYS_EMPLOYED"]>0).mean()))
print ('# of positive values: {}'.format((application_train["DAYS_EMPLOYED"]>0).sum()))

#%%
application_train["DAYS_EMPLOYED"] = application_train["DAYS_EMPLOYED"].replace(365243, np.nan)
display(application_train["DAYS_EMPLOYED"].value_counts())
print ('% of positive values: {:.0%}'.format((application_train["DAYS_EMPLOYED"]>0).mean()))
print ('# of positive values: {}'.format((application_train["DAYS_EMPLOYED"]>0).sum()))

#%%
## FV1
application_train['income_per_person'] = application_train['AMT_INCOME_TOTAL'] / application_train['CNT_FAM_MEMBERS']

## FV2
application_train['income_per_employed'] = application_train['AMT_INCOME_TOTAL'] / application_train['DAYS_EMPLOYED']

## FV3
application_train['ext_source_mean'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
application_train['ext_source_max'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
application_train['ext_source_min'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
application_train['ext_source_std'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
application_train['ext_source_count'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notnull().sum(axis=1)

## FV4
application_train['days_employed_per_birth'] = application_train['DAYS_EMPLOYED'] / application_train['DAYS_BIRTH']

## FV5
application_train['annuity_per_income'] = application_train['AMT_ANNUITY'] / application_train['AMT_INCOME_TOTAL']

## FV6
application_train['annuity_per_credit'] = application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']

#%%
def make_training_dataset(application_train):
    x_train = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])
    y_train = application_train['TARGET']
    id_train = application_train['SK_ID_CURR']

    for col in x_train.columns:
        if x_train[col].dtype == 'O':
            x_train[col] = x_train[col].astype('category')

    return x_train, y_train, id_train

def make_test_dataset(application_test):
    x_test = application_test.drop(columns=['SK_ID_CURR'])
    id_test = application_test['SK_ID_CURR']

    for col in x_test.columns:
        if x_test[col].dtype == 'O':
            x_test[col] = x_test[col].astype('category')

    return x_test, id_test

#%%
x_train, y_train, id_train = make_training_dataset(application_train)

train_oof, imp, metrics = train_lgb(
    x_train, y_train, id_train, params, list_nfold=[0,1,2,3,4], n_splits=5
)

#%%
imp.sort_values('imp', ascending=False)[:10]

#%%
application_test["DAYS_EMPLOYED"] = application_test["DAYS_EMPLOYED"].replace(365243, np.nan)

## FV1
application_test['income_per_person'] = application_test['AMT_INCOME_TOTAL'] / application_test['CNT_FAM_MEMBERS']

## FV2
application_test['income_per_employed'] = application_test['AMT_INCOME_TOTAL'] / application_test['DAYS_EMPLOYED']

## FV3
application_test['ext_source_mean'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
application_test['ext_source_max'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
application_test['ext_source_min'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
application_test['ext_source_std'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
application_test['ext_source_count'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notnull().sum(axis=1)

## FV4
application_test['days_employed_per_birth'] = application_test['DAYS_EMPLOYED'] / application_test['DAYS_BIRTH']

## FV5
application_test['annuity_per_income'] = application_test['AMT_ANNUITY'] / application_test['AMT_INCOME_TOTAL']

## FV6
application_test['annuity_per_credit'] = application_test['AMT_ANNUITY'] / application_test['AMT_CREDIT']

#%%
x_test, id_test = make_test_dataset(application_test)
test_pred = predict_lgb(x_test, id_test)
df_submit = test_pred.rename(columns={'pred': 'TARGET'})
print(df_submit.shape)
display(df_submit.head())
df_submit.to_csv('../output/submission_FeatureEngineering.csv', index=None)

#%%
pos = pd.read_csv('../input/POS_CASH_balance.csv')
pos = reduce_mem_usage(pos)
print (pos.shape)
pos.head()

#%%
pos_ohe = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
col_ohe = sorted(list(set(pos_ohe.columns) - set(pos.columns)))
print(len(col_ohe))
display(col_ohe)

#%%
pos_ohe_agg = pos_ohe.groupby('SK_ID_CURR').agg({
    'MONTHS_BALANCE': ['mean', 'std', 'min', 'max'],
    'CNT_INSTALMENT': ['mean', 'std', 'min', 'max'],
    'CNT_INSTALMENT_FUTURE': ['mean', 'std', 'min', 'max'],
    'SK_DPD': ['mean', 'std', 'min', 'max'],
    'SK_DPD_DEF': ['mean', 'std', 'min', 'max'],
    'NAME_CONTRACT_STATUS_Active': ['mean'],
    'NAME_CONTRACT_STATUS_Amortized debt': ['mean'],
    'NAME_CONTRACT_STATUS_Approved': ['mean'],
    'NAME_CONTRACT_STATUS_Canceled': ['mean'],
    'NAME_CONTRACT_STATUS_Completed': ['mean'],
    'NAME_CONTRACT_STATUS_Demand': ['mean'],
    'NAME_CONTRACT_STATUS_Returned to the store': ['mean'],
    'NAME_CONTRACT_STATUS_Signed': ['mean'],
    'NAME_CONTRACT_STATUS_XNA': ['mean'],
    'NAME_CONTRACT_STATUS_nan': ['mean'],
    'SK_ID_PREV': ['count', 'nunique']
})

pos_ohe_agg.columns = [i + '_' + j for i,j in pos_ohe_agg.columns]
pos_ohe_agg = pos_ohe_agg.reset_index(drop=False)

print (pos_ohe_agg.shape)
display(pos_ohe_agg.head())

#%%
df_train = pd.merge(application_train, pos_ohe_agg, on='SK_ID_CURR', how='left')
print(df_train.shape)
df_train.head()

#%%
x_train, y_train, id_train = make_training_dataset(df_train)
train_oof, imp, metrics = train_lgb(
    x_train, y_train, id_train, params, list_nfold=[0,1,2,3,4],n_splits=5
)

#%%
imp.sort_values('imp', ascending=False)[:10]

#%%
df_test = pd.merge(application_test, pos_ohe_agg, on='SK_ID_CURR', how='left')
x_test, id_test = make_test_dataset(df_test)

test_pred = predict_lgb(
    x_test, id_test
)

#%%
df_submit = test_pred.rename(columns={'pred': 'TARGET'})
print(df_submit.shape)
display(df_submit.head())
df_submit.to_csv('../output/submission_FeatureEngineering2.csv', index=None)

#%%
import optuna

x_train, y_train, id_train = make_training_dataset(df_train)
params_base = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1, 
    'learning_rate': 0.05,
    'n_estimators': 100000,
    'bagging_freq': 1
}

def objective(trial):
    params_tuning = {
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-5, 1e-2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-2, 1e+2, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 1e+2, log=True),
    }
    params_tuning.update(params_base)

    list_metrics = []
    cv = list(StratifiedKFold(n_splits=5,shuffle=True,random_state=123).split(x_train, y_train))
    list_fold = [0]
    for nfold in list_fold:
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr]
        x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)],
            early_stopping_rounds=100, verbose=0
        )
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_va = roc_auc_score(y_va, y_va_pred)
        list_metrics.append(metric_va)
    
    metrics = np.mean(list_metrics)

    return metrics

#%%
sampler = optuna.samplers.TPESampler(seed=123)
study = optuna.create_study(sampler=sampler, direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=5)

#%%
trial = study.best_trial
print ('auc(best) = {:.4f}'.format(trial.value))
display(trial.params)

params_best = trial.params
params_best.update(params_base)
display(params_best)

#%%
train_oof, imp, metrics = train_lgb(
    x_train, y_train, id_train, params=params_best
)

#%%
x_test, id_test = make_test_dataset(df_test)
test_pred = predict_lgb(
    x_test, id_test
)

df_submit = test_pred.rename(columns={'pred': 'TARGET'})
print(df_submit.shape)
display(df_submit.head())
df_submit.to_csv('../output/submission_HyperParameterTuning.csv', index=None)