# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts

import lightgbm as lgb
# -

# # Data Processing

# +
seed = 0

train_df = pd.read_csv("../input/train.csv", index_col="text_id")
X_train = train_df.full_text
cols = [col for col in train_df.columns if col != "full_text"]
y_train = train_df[cols]
X_test = pd.read_csv("../input/test.csv", index_col="text_id").full_text
X_test_idx = X_test.index

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train, "tfidf")
X_test = tokenizer.texts_to_matrix(X_test, "tfidf")

pca = PCA(n_components=100, whiten=True, random_state=seed)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=.1, random_state=seed)

lgb_trains = {}
lgb_vals = {}
for col in cols:
    exec(f"lgb_trains['{col}'] = lgb.Dataset(X_train, y_train.{col})")
    exec(f"lgb_vals['{col}'] = lgb.Dataset(X_val, y_val.{col})")


# -

# # Custom Loss Function - MCRMSE

# +
@tf.autograph.experimental.do_not_convert
def MCRMSE_keras(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))

def MCRMSE_lgb(preds, eval_data):
    diff = eval_data - preds
    sq = np.square(diff)
    rmse = np.sum(sq, axis=0) / eval_data.shape[0]
    return "MCRMSE", np.sum(rmse) / eval_data.shape[1], False


# -

# # Build Models

# ## keras 1

# +
keras1_model = Sequential()
keras1_model.add(Dense(500, input_dim=X_train.shape[1], activation="relu"))
keras1_model.add(BatchNormalization())
keras1_model.add(Dense(500, activation="relu"))
keras1_model.add(Dropout(.3))
keras1_model.add(Dense(500, activation=LeakyReLU(.1)))
keras1_model.add(Dropout(.2))
keras1_model.add(Dense(500, activation="relu"))
keras1_model.add(Dense(y_train.shape[1], activation="sigmoid"))
keras1_model.add(Rescaling(4, offset=1))

optimizer = optimizers.Adam(amsgrad=True)
keras1_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])
keras1_model.fit(X_train, y_train, batch_size=2**3, epochs=30, verbose=1,
          validation_data=(X_val, y_val), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])

keras1_pred = pd.DataFrame(keras1_model.predict(X_test), columns=cols, index=X_test_idx)
keras1_pred
# -

# ## keras 2

# +
keras2_model = Sequential()
keras2_model.add(Dense(2000, input_dim=X_train.shape[1], activation="relu"))
keras2_model.add(BatchNormalization())
keras2_model.add(Dense(2000, activation="relu"))
keras2_model.add(Dropout(.3))
keras2_model.add(Dense(3000, activation=LeakyReLU(.1)))
keras2_model.add(Dropout(.2))
keras2_model.add(Dense(2000, activation="relu"))
keras2_model.add(Dense(500, activation="relu"))
keras2_model.add(Dense(3000, activation="softplus"))
keras2_model.add(BatchNormalization())
keras2_model.add(Dense(1000, activation=LeakyReLU(.1)))
keras2_model.add(Dropout(.3))
keras2_model.add(Dense(3000, activation="softsign"))
keras2_model.add(Dense(1000, activation=LeakyReLU(.1)))
keras2_model.add(Dropout(.1))
keras2_model.add(Dense(3000, activation="softplus"))
keras2_model.add(Dropout(.4))
keras2_model.add(Dense(3000, activation="relu"))
keras2_model.add(Dense(1000, activation="relu"))
keras2_model.add(BatchNormalization())
keras2_model.add(Dense(y_train.shape[1], activation="sigmoid"))
keras2_model.add(Rescaling(4, offset=1))

optimizer = optimizers.Adam(amsgrad=True)
keras2_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])
keras2_model.fit(X_train, y_train, batch_size=2**3, epochs=50, verbose=1,
          validation_data=(X_val, y_val), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])

keras2_pred = pd.DataFrame(keras2_model.predict(X_test), columns=cols, index=X_test_idx)
keras2_pred
# -

# ## lgb 1

# +
lgb1_models = {}
lgb1_preds = {}

for score in cols:
    lgb1_params = {'objective': 'regression',
                   'metric': 'rmse',
                   'verbosity': 0,
                   'early_stopping_round': 50,
                   'random_state': seed,
                   'device': 'gpu'}
    
    train_set=lgb_trains[score]
    valid_sets=lgb_vals[score]

    lgb1_model = lgb.train(
        params=lgb1_params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=(train_set, valid_sets),
        callbacks=None,
        verbose_eval=100
    )
    
    lgb1_models[score] = lgb1_model
    lgb1_preds[score] = lgb1_model.predict(X_test)
    
lgb1_pred = pd.DataFrame(lgb1_preds, index=X_test_idx)
lgb1_pred
# -

pred = pd.DataFrame(np.mean(np.array([keras1_pred, keras2_pred, lgb1_pred]), axis=0), columns=cols, index=X_test_idx)

pred.to_csv("submission.csv", index=True)


