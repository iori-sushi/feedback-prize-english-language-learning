# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1. Preparation

# ## 1.1. Import Libraries

# +
import os
import re
import string
import warnings
warnings.simplefilter("ignore")

import lightgbm as lgb
import nltk
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from keras.layers import add, concatenate
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from optuna.integration import KerasPruningCallback
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as tts
from sklearn.utils.class_weight import compute_class_weight as ccw
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Rescaling,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed, to_categorical

pd.options.display.max_columns=100
nltk.download('stopwords')
# -

# ## 1.2. Fetching Data

# +
seed = 0
np.random.seed(seed)
set_random_seed(seed)

train_df = pd.read_csv("../input/train.csv", index_col="text_id")
X_train = train_df.full_text
cols = [col for col in train_df.columns if col != "full_text"]
y_train = train_df[cols]
X_test = pd.read_csv("../input/test.csv", index_col="text_id").full_text
X_test_idx = X_test.index


# -

# ## 1.3. Custom Fanctions

# memo
# encoding, onebyone_wrapper, kfold, oversampling, building_model, check_results
# optuna, optunaresult, predict

@tf.autograph.experimental.do_not_convert
def MCRMSE(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))


class BuildModel:
    
    def __init__(self, X_train = X_train, y_train = y_train, X_test = X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train

    def execution(self, params):
        X_train_enc, X_test = self.encodingX(params["pca_components"])
        preds = []; evals = []
        X_trains, X_vals, y_trains, y_vals = self.kfold(X_train_enc, self.y_train, n_splits=params["valid_splits"])
        for X_train, X_val, y_train, y_val in zip(X_trains, X_vals, y_trains, y_vals):
            X_train, y_train = self.oversampling(X_train, y_train)
            model, evaluation = self.modeling(X_train, y_train, params, X_val, y_val)
            preds += [pd.DataFrame(model.predict(X_val), index=X_val.index, columns=[y_train.name])]
            evals += [evaluation]
        return np.mean(evals), pd.concat(preds, axis=0)
    
    def finalize(self, y_train, params):
        X_train_enc, X_test_enc = self.encodingX(params["pca_components"])
        X_train, y_train = self.oversampling(X_train_enc, y_train)
        model = self.modeling(X_train, y_train, params)
        pred = pd.DataFrame(model.predict(X_test_enc), index=X_test_enc.index, columns=[y_train.name])
        return pred, model
    
    def encodingX(self, n_components = 500):
        train_idx = self.X_train.index
        test_idx = self.X_test.index

        X_train = self.X_train.str.lower().replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
        X_test = self.X_test.str.lower().replace(re.compile(r'[\n\r\t]'), ' ', regex=True)

        stop_words = stopwords.words('english')
        remove_stopwords = lambda ls: [" ".join([word for word in l.split() if not word in stop_words]) for l in ls] 
        X_train = remove_stopwords(X_train)
        X_test = remove_stopwords(X_test)

        X_train = [re.sub("[%s]" % re.escape(string.punctuation), "", t) for t in X_train]
        X_test = [re.sub("[%s]" % re.escape(string.punctuation), "", t) for t in X_test]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_matrix(X_train, "tfidf")
        X_test = tokenizer.texts_to_matrix(X_test, "tfidf")

        pca = PCA(n_components=n_components, whiten=True, random_state=seed)
        X_train = pd.DataFrame(pca.fit_transform(X_train), index=train_idx)
        X_test = pd.DataFrame(pca.transform(X_test), index=test_idx)
        return X_train, X_test
    
    def kfold(self, X_train, y_train, n_splits = 4):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X_trains = []; X_vals = []; y_trains = []; y_vals = []
        for train_idx, val_idx in skf.split(X_train, y_train.astype(str)):
            X_trains.append(X_train.iloc[train_idx,:])
            X_vals.append(X_train.iloc[val_idx,:])
            y_trains.append(y_train.iloc[train_idx])
            y_vals.append(y_train.iloc[val_idx])
        return X_trains, X_vals, y_trains, y_vals
        
    def oversampling(self, X_train, y_train):
        X_idx = list(X_train.index)
        maximum = int(len(y_train) * 1.5)
        agg = y_train.value_counts()
        mean = int(agg.mean())
        agg_ltm = agg < mean
        strategy_value = (maximum - (agg[agg>=mean]).sum()) // (agg_ltm).sum()
        add_value = (maximum - (agg[agg>=mean]).sum()) % (agg_ltm).sum()
        sampling_strategy = {str(k): strategy_value + add_value if k == agg[agg_ltm].index[0] else strategy_value for k in agg_ltm[agg_ltm].index}
        X_train, y_train = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X_train, y_train.astype(str))
        idx = X_idx + list(X_train.index[len(X_idx):])
        X_train.index = idx
        y_train.index = idx
        y_train = y_train.astype(np.float32)
        return X_train, y_train

    def modeling(self, X_train, y_train, params, X_val = None, y_val = None):
        model = Sequential()
        for i in range(1, params["NLayers"]+1):
            if params[f"BatchNorm_l{i}"]:
                model.add(BatchNormalization())
            if params[f"Activation_l{i}"] == "lrelu":
                params[f"Activation_l{i}"] = LeakyReLU(alpha=.1)
            if i == 1:
                model.add(Dense(params[f"Units_l{i}"], input_dim=X_train.shape[1], activation=params[f"Activation_l{i}"]))
            else:
                model.add(Dense(params[f"Units_l{i}"], activation=params[f"Activation_l{i}"]))
            if params[f"Dropout_l{i}"]:
                model.add(Dropout(.3))
        if params["Rescaling"]:
            model.add(Dense(1, activation="sigmoid"))
            model.add(Rescaling(4, offset=1))
        else:
            model.add(Dense(1, activation="linear"))

        optimizer = optimizers.Adam(learning_rate=params["LearningRate"], amsgrad=True)
        model.compile(loss=MCRMSE, optimizer=optimizer, metrics=MCRMSE)

        if not X_val is None:
            callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
            if not params["Pruner"] is None:
                callbacks += [params["Pruner"]]
        
            model.fit(X_train, y_train, validation_data = (X_val, y_val),
                      batch_size=params["BatchSize"], epochs=100,
                      workers=8, use_multiprocessing=True, verbose=0, callbacks=callbacks)
            evaluation = model.evaluate(X_val, y_val, verbose=0)
            return model, evaluation
        else:
            model.fit(X_train, y_train,
                      batch_size=params["BatchSize"], epochs=100,
                      workers=8, use_multiprocessing=True, verbose=0)
            return model


# + tags=[] jupyter={"outputs_hidden": true, "source_hidden": true}
res = study_result("keras1")
params = {k:v for k,v in zip(res.columns, res.values[0])} | {"BatchSize": 2**3, "Rescaling": False, "Pruner": None, "valid_splits":4, "pca_components": 500}
build = BuildModel(y_train = y_train.syntax)
res = build.execution(params)
res


# -

class ParamTuning:
    
    def __init__(self,
                 y_train,
                 storage='mysql+pymysql://iori:'+os.environ["MySQL_PASSWORD"]+'@'+os.environ["MySQL_IP"]+":3306/kaggleELL?charset=utf8mb4",
                 directions=["minimize"]):
        self.y_train = y_train
        self.storage = storage
        self.directions = directions
        self.study_name = "keras-" + y_train.name

    def execution(self, trial):
        params = self.get_param(trial)
        evals, pred = BuildModel(y_train=self.y_train).execution(params)
        return evals
    
    def get_param(self, trial):
        param = {
            "valid_splits": 4,
            "pca_components": 500,
            "NLayers": trial.suggest_int("NLayers", 7, 20),
            "LearningRate": trial.suggest_uniform("LearningRate", .0005, .01),
            "BatchSize": 2**3,
            'Rescaling': False,#trial.suggest_categorical('Rescaling', [True, False]) 
            "Pruner": None #KerasPruningCallback(trial, "val_acc"),
            }

        for i in range(1, param["NLayers"]+1):
            param[f"BatchNorm_l{i}"] = trial.suggest_categorical(f'BatchNorm_l{i}', [True, False])
            param[f"Units_l{i}"] = int(trial.suggest_uniform(f"Units_l{i}", 300, 2000))
            activation = trial.suggest_categorical(f'Activation_l{i}', ['relu', "lrelu", "softplus", "softsign"])
            if activation == "lrelu":
                activation = LeakyReLU(alpha=.1)
            param[f'Activation_l{i}'] = activation
            param[f'Dropout_l{i}'] = trial.suggest_categorical(f'Dropout_l{i}', [True, False]) 
        return param        

    def studying(self, n_trials=1):
        tf.keras.backend.clear_session()
        keras_study = optuna.create_study(study_name=self.study_name,
                                          directions=self.directions,# pruner=optuna.pruners.MedianPruner(), 
                                          storage=self.storage,
                                          load_if_exists=True)
        keras_study.optimize(self.execution, n_trials=n_trials, gc_after_trial=True)
  
    def show_result(self):
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        bts = []
        for bt in study.best_trials:
            values = bt.values or [0]
            values = [bt.number] + values + [bt.datetime_complete - bt.datetime_start] + list(bt.params.values())
            bts.append(pd.DataFrame([values], columns = ["number", "MCRMSE", "exec_time"] + list(bt.params.keys())))
            res = pd.concat(bts, axis=0, ignore_index=True).sort_values("MCRMSE")
        return res


res = {}
for target in y_train.columns:
    res[target] = ParamTuning(y_train=y_train[target]).show_result()

for target in y_train.columns:
    res[target] = {k:v for k, v in zip(res[target].columns, res[target].values[0])} | {"BatchSize": 2**3,
 "Rescaling": False, 
 "Pruner": None,
 "pca_components": 500}
res

for target in y_train.columns[2:]:
    param_tuning = ParamTuning(y_train = y_train[target])
    param_tuning.studying(n_trials=5)
    res = param_tuning.show_result()
    display(res)


# + tags=[]
def prediction(X_test, model):
    return pd.DataFrame(model.predict(X_test), columns=cols, index=X_test_idx)

def concat_models(X_train, y_train, models, X_val=None, y_val=None):
    model = concatenate([model.output for model in models])
    model = Dense(500, activation="relu")(model)
    model = Dense(1000, activation="relu")(model)
    model = Dense(500, activation="relu")(model)
    model = Dense(y_train.iloc[:,:2].shape[1], activation="linear")(model)
    model = Model([model.input for model in models], model)
    
    optimizer = optimizers.Adam(amsgrad=True)
    model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])
    callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    model.fit([X_train, X_train], y_train.iloc[:,:2],# validation_data = (X_val, y_val),
              batch_size=2**3, epochs=10,
              workers=8, use_multiprocessing=True, verbose=1, callbacks=callbacks)
    
    if not X_val is None:
        evaluation = model.evaluate(X_val, y_val, verbose=0)
        return model, evaluation
    else:
        return model


# + [markdown] tags=[]
# # 2. Building Models

# + [markdown] tags=[]
# ## 2.0. set the variables
# -

storage = 'mysql+pymysql://iori:'+os.environ["MySQL_PASSWORD"]+'@'+os.environ["MySQL_IP"]+":3306/kaggleELL?charset=utf8mb4"
directions = ["minimize"]

# + [markdown] tags=[]
# ## 2.1. keras1

# +
study_name = "keras1"

keras_study = optuna.create_study(study_name=study_name, directions=directions,# pruner=optuna.pruners.MedianPruner(), 
                                  storage=storage, load_if_exists=True)
keras_study.optimize(objective, n_trials=50, gc_after_trial=True)
res = study_result(study_name)
res

# +
models = []

res = study_result("keras1")
params = {k:v for k,v in zip(res.columns, res.values[0])} | {"BatchSize": 2**3, "Rescaling": False, "Pruner": None}
X_train_enc, X_test_enc = encodingX()
for col in cols[:2]:
    models.append(building_keras_model(X_train_enc, y_train[col], params))
model = concat_models(X_train_enc, y_train, models, X_val=None, y_val=None)
model.predict([X_test_enc] * 2)
# -

model = concat_models(X_train_enc, y_train, models, X_val=None, y_val=None)
model.predict(X_test_enc)

model.predict([X_test_enc, X_test_enc])

import seaborn as sns
import matplotlib.pyplot as plt
count = 1
fig = plt.figure(figsize=(25,10))
for col in cols:
    for score in sorted(y_train[col].unique()):
        ax = fig.add_subplot(6, 9, count)
        print(col, score)
        data = pd.concat([y_train[col], pred[col]], axis=1)
        data.columns = ["true", "pred"]
        sns.distplot(data[data["true"]==score].pred, ax = ax, kde=False);
        ax.set_xlabel(score);
        ax.set_xlim([1,5]);
        ax.set_title(col+str(score));
        plt.tight_layout()
        fig.show();
        count += 1

preds = [keras1_pred, keras2_pred, keras3_pred, keras4_pred, lgb1_pred, lgb2_pred]
pred = pd.DataFrame(np.mean(np.array(preds), axis=0), columns=cols, index=X_test_idx)
pred

pred.to_csv("submission.csv", index=True)
