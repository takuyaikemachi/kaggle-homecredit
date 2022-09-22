#%%
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
nfold = 0
idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
print (idx_tr)
x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
#x_tr, y_tr, id_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :], id_train.loc[idx_tr, :]
#x_va, y_va, id_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :], id_train.loc[idx_va, :]
#print (x_tr.shape, x_va.shape)


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