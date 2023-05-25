# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1. Preparation

# ## 1.1. Import Libraries

# +
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling2D,
    Rescaling,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed, to_categorical
from transformers import (
    RobertaTokenizer,
    TFRobertaModel,
)

warnings.simplefilter("ignore")
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

# ## 1.3. Sentence Encodings

# ### 1.3.0. Custom Functions

# +
def bert_encoder(texts, tokenizer, max_len):
    input_ids = []; attention_mask = []
    for text in texts:
        token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length',
                         add_special_tokens=True, return_attention_mask=True)
        input_ids.append(token['input_ids'])
        attention_mask.append(token['attention_mask'])    
    return np.array(input_ids), np.array(attention_mask)

def train_test_split_scratch(X, y, test_size=.1):
    all_num = X[0].shape[0]
    test_num = int(all_num * (test_size))
    test_idx = np.random.choice(all_num, test_num, replace=False)
    train_idx = np.setdiff1d(np.arange(all_num), test_idx)
    return (X[0][train_idx], X[1][train_idx]), (X[0][test_idx], X[1][test_idx]), y.iloc[train_idx], y.iloc[test_idx]

reshape_func = lambda df: df.reshape(list(df.shape)+[1])
# -

# ### 1.3.1. TFIDF with PCA

# +
tokenizer_tfidf = Tokenizer()
tokenizer_tfidf.fit_on_texts(X_train)
X_train_tfidf = tokenizer_tfidf.texts_to_matrix(X_train, "tfidf")
X_test_tfidf = tokenizer_tfidf.texts_to_matrix(X_test, "tfidf")

pca_tfidf = PCA(n_components=500, whiten=True, random_state=seed)
X_train_tfidf = pca_tfidf.fit_transform(X_train_tfidf)
X_test_tfidf = pca_tfidf.transform(X_test_tfidf)

X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = tts(X_train_tfidf, y_train, test_size=.1, random_state=seed)

lgb_trains_tfidf = {}
lgb_vals_tfidf = {}
for col in cols:
    exec(f"lgb_trains_tfidf['{col}'] = lgb.Dataset(X_train_tfidf, y_train_tfidf.{col})")
    exec(f"lgb_vals_tfidf['{col}'] = lgb.Dataset(X_val_tfidf, y_val_tfidf.{col})")
# -

# ### 1.3.2. roberta-base

# +
max_len = 512
roberta_dir = "roberta-base"#'../input/roberta-base/'

tokenizer_roberta = RobertaTokenizer.from_pretrained(roberta_dir)
model_roberta = TFRobertaModel.from_pretrained(roberta_dir)

X_train_roberta = bert_encoder(X_train, tokenizer_roberta, max_len)
X_test_roberta = bert_encoder(X_test, tokenizer_roberta, max_len)
X_train_roberta, X_val_roberta, y_train_roberta, y_val_roberta = train_test_split_scratch(X_train_roberta, y_train)

input_ids_roberta  = Input(shape=(X_train_roberta[0].shape[1], ), dtype = tf.int32)
attention_mask_roberta = Input(shape=(X_train_roberta[1].shape[1], ), dtype = tf.int32)
roberta_ = model_roberta(input_ids = input_ids_roberta, attention_mask = attention_mask_roberta)
roberta_ = Model(inputs = [input_ids_roberta, attention_mask_roberta], outputs = roberta_)
roberta_.compile(loss="rmse", optimizer="rmsprop")
X_train_roberta_3d = reshape_func(roberta_.predict(X_train_roberta).last_hidden_state)
X_val_roberta_3d = reshape_func(roberta_.predict(X_val_roberta).last_hidden_state)
X_test_roberta_3d = reshape_func(roberta_.predict(X_test_roberta).last_hidden_state)

pca_roberta = PCA(n_components=3000, whiten=True, random_state=seed)
X_train_roberta_pca = pca_roberta.fit_transform(X_train_roberta_3d.reshape((X_train_roberta_3d.shape[0], -1)))
X_val_roberta_pca = pca_roberta.transform(X_val_roberta_3d.reshape((X_val_roberta_3d.shape[0], -1)))
X_test_roberta_pca = pca_roberta.transform(X_test_roberta_3d.reshape((X_test_roberta_3d.shape[0], -1)))

lgb_trains_roberta = {}
lgb_vals_roberta = {}
for col in cols:
    exec(f"lgb_trains_roberta['{col}'] = lgb.Dataset(X_train_roberta_pca, y_train_roberta.{col})")
    exec(f"lgb_vals_roberta['{col}'] = lgb.Dataset(X_val_roberta_pca, y_val_roberta.{col})")

# + tags=[] jupyter={"outputs_hidden": true}
# %%time

tokenizer_tfidf = Tokenizer()
tokenizer_tfidf.fit_on_texts(X_train)
X_train_tfidf = tokenizer_tfidf.texts_to_matrix(X_train, "tfidf")
print(X_train_tfidf.shape)

pca_tfidf = PCA(n_components=3000, whiten=True, random_state=seed)
print(pca_tfidf.fit_transform(X_train_tfidf).__sizeof__())

plt.plot(np.cumsum(pca_tfidf.explained_variance_ratio_))
plt.show()

# + tags=[] jupyter={"outputs_hidden": true}
# %%time

print(X_train_tfidf.shape)

svd_tfidf = TruncatedSVD(n_components=3000, random_state=seed)
print(svd_tfidf.fit_transform(X_train_tfidf).__sizeof__())

plt.plot(np.cumsum(svd_tfidf.explained_variance_ratio_))
plt.show()

# + jupyter={"outputs_hidden": true} tags=[]
# %%time
pca_roberta = PCA(n_components=3000, random_state=seed)
check = pca_roberta.fit_transform(X_train_roberta_3d.reshape((X_train_roberta_3d.shape[0],-1)))
check.shape
print(check.__sizeof__())

import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca_roberta.explained_variance_ratio_))
plt.show()

# + tags=[] jupyter={"outputs_hidden": true}
# %%time
from sklearn.decomposition import TruncatedSVD
svd_roberta = TruncatedSVD(n_components=3000, random_state=seed)
check2 = svd_roberta.fit_transform(X_train_roberta_3d.reshape((X_train_roberta_3d.shape[0],-1)))
check2.shape
print(check2.__sizeof__())

plt.plot(np.cumsum(svd_roberta.explained_variance_ratio_))
plt.show()


# + [markdown] tags=[]
# # 2. Building Models
# -

# ## 2.0. Custom Loss Functions

# +
@tf.autograph.experimental.do_not_convert
def MCRMSE_keras(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))

def MCRMSE_lgb(preds, eval_data):
    diff = eval_data - preds
    sq = np.square(diff)
    rmse = np.sum(sq, axis=0) / eval_data.shape[0]
    return "MCRMSE", np.sum(rmse) / eval_data.shape[1], False


# + [markdown] tags=[]
# ## 2.1. keras1: small Keras Model with tfidf-encoding

# + tags=[]
keras1_model = Sequential()
keras1_model.add(Dense(500, input_dim=X_train_tfidf.shape[1], activation="relu"))
keras1_model.add(BatchNormalization())
keras1_model.add(Dense(500, activation="relu"))
keras1_model.add(Dropout(.5))
keras1_model.add(Dense(500, activation="relu"))
keras1_model.add(Dropout(.5))
keras1_model.add(Dense(200, activation="relu"))
keras1_model.add(Dense(y_train.shape[1], activation="sigmoid"))
keras1_model.add(Rescaling(4, offset=1))

optimizer = optimizers.Adam(amsgrad=True)
keras1_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])
keras1_model.fit(X_train_tfidf, y_train_tfidf, batch_size=2**3, epochs=100, verbose=1,
          validation_data=(X_val_tfidf, y_val_tfidf), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras1_pred = pd.DataFrame(keras1_model.predict(X_test_tfidf), columns=cols, index=X_test_idx)
keras1_pred

# + tags=[]
keras1_model = Sequential()
keras1_model.add(Dense(500, input_dim=X_train_tfidf.shape[1], activation="relu"))
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
keras1_model.fit(X_train_tfidf, y_train_tfidf, batch_size=2**3, epochs=30, verbose=1,
          validation_data=(X_val_tfidf, y_val_tfidf), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras1_pred = pd.DataFrame(keras1_model.predict(X_test_tfidf), columns=cols, index=X_test_idx)
keras1_pred
# -

# ## 2.2. keras2: large Keras Model with tfidf-encoding

# + tags=[]
keras2_model = Sequential()
keras2_model.add(Dense(2000, input_dim=X_train_tfidf.shape[1], activation="relu"))
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
keras2_model.fit(X_train_tfidf, y_train_tfidf, batch_size=2**3, epochs=50, verbose=1,
          validation_data=(X_val_tfidf, y_val_tfidf), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras2_pred = pd.DataFrame(keras2_model.predict(X_test_tfidf), columns=cols, index=X_test_idx)
keras2_pred

# + tags=[]
keras2_model = Sequential()
keras2_model.add(Dense(2000, input_dim=X_train_tfidf.shape[1], activation="relu"))
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
keras2_model.fit(X_train_tfidf, y_train_tfidf, batch_size=2**3, epochs=50, verbose=1,
          validation_data=(X_val_tfidf, y_val_tfidf), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras2_pred = pd.DataFrame(keras2_model.predict(X_test_tfidf), columns=cols, index=X_test_idx)
keras2_pred
# -

# ## 2.3. keras3: small Keras Model with roberta-encoding (2d input)

# +
keras3_model = Sequential()
keras3_model.add(Dense(500, input_dim=X_train_roberta_pca.shape[1],  activation="relu"))
keras3_model.add(Dense(1000, activation="relu"))
keras3_model.add(Dropout(.5))
keras3_model.add(Dense(500, activation="relu"))
keras3_model.add(Dense(y_train_roberta.shape[1], activation="sigmoid"))
keras3_model.add(Rescaling(4, offset=1))

optimizer = optimizers.Adam(amsgrad=True)
keras3_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])

keras3_model.fit(X_train_roberta_pca, y_train_roberta, batch_size=2**3, epochs=30, verbose=1,
          validation_data=(X_val_roberta_pca, y_val_roberta), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras3_pred = pd.DataFrame(keras3_model.predict(X_test_roberta_pca), columns=cols, index=X_test_idx)
keras3_pred
# -

# ## 2.4. keras4: small CNN Keras Model with roberta-encoding (3d input)

# +
keras4_model = Sequential()
keras4_model.add(BatchNormalization())
keras4_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train_roberta_3d.shape[1:]))
keras4_model.add(MaxPooling2D((2, 2)))
keras4_model.add(Conv2D(64, (3, 3), activation='relu'))
keras4_model.add(MaxPooling2D((2, 2)))
keras4_model.add(Conv2D(64, (3, 3), activation='relu'))
keras4_model.add(Flatten())
keras4_model.add(BatchNormalization())
keras4_model.add(Dense(500,  activation="relu"))
keras4_model.add(Dense(1000, activation="relu"))
keras4_model.add(Dropout(.1))
keras4_model.add(Dense(500, activation="relu"))
keras4_model.add(Dense(y_train_roberta.shape[1], activation="linear"))
#keras4_model.add(Rescaling(4, offset=1))z

optimizer = optimizers.Adam(amsgrad=True)
keras4_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])

keras4_model.fit(X_train_roberta_3d, y_train_roberta, batch_size=2**3, epochs=30, verbose=1,
          validation_data=(X_val_roberta_3d, y_val_roberta), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras4_pred = pd.DataFrame(keras4_model.predict(X_test_roberta_3d), columns=cols, index=X_test_idx)
keras4_pred
# -

# ## 2.5. keras5: small Keras Model with roberta-encoding (2d input - 1 choosed layer)

# +
keras5_target = 1

keras5_model = Sequential()
keras5_model.add(Dense(500, input_dim=X_train_roberta_3d[:,keras5_target,:,0].shape[1],  activation="relu"))
keras5_model.add(Dense(1000, activation="relu"))
keras5_model.add(Dropout(.5))
keras5_model.add(Dense(500, activation="relu"))
keras5_model.add(Dense(y_train_roberta.shape[1], activation="sigmoid"))
keras5_model.add(Rescaling(4, offset=1))

optimizer = optimizers.Adam(amsgrad=True)
keras5_model.compile(loss=MCRMSE_keras, optimizer=optimizer, metrics=[MCRMSE_keras])

keras5_model.fit(X_train_roberta_3d[:,keras5_target,:,0], y_train_roberta, batch_size=2**3, epochs=30, verbose=1,
          validation_data=(X_val_roberta_3d[:,keras5_target,:,0], y_val_roberta), workers=30, use_multiprocessing=True,
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

keras5_pred = pd.DataFrame(keras5_model.predict(X_test_roberta_3d[:,keras5_target,:,0]), columns=cols, index=X_test_idx)
keras5_pred
# -

# ## 2.4. lgb1: LightGBM Model with tfidf-encoding

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
    
    train_set=lgb_trains_tfidf[score]
    valid_sets=lgb_vals_tfidf[score]

    lgb1_model = lgb.train(
        params=lgb1_params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=(train_set, valid_sets),
        callbacks=None,
        verbose_eval=100
    )
    
    lgb1_models[score] = lgb1_model
    lgb1_preds[score] = lgb1_model.predict(X_test_tfidf)
    
lgb1_pred = pd.DataFrame(lgb1_preds, index=X_test_idx)
lgb1_pred
# -

# ## 2.5. lgb2: LightGBM Model with roberta_ids

# +
lgb2_models = {}
lgb2_preds = {}

for score in cols:
    lgb2_params = {'objective': 'regression',
                   'metric': 'rmse',
                   'verbosity': 0,
                   'early_stopping_round': 50,
                   'random_state': seed,
                   'device': 'gpu'}
    
    train_set=lgb_trains_roberta[score]
    valid_sets=lgb_vals_roberta[score]

    lgb2_model = lgb.train(
        params=lgb2_params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=(train_set, valid_sets),
        callbacks=None,
        verbose_eval=100
    )
    
    lgb2_models[score] = lgb2_model
    lgb2_preds[score] = lgb2_model.predict(X_test_roberta_3d[:,1,:,0])
    
lgb2_pred = pd.DataFrame(lgb2_preds, index=X_test_idx)
lgb2_pred
# -

# ## 2.5. lgb3: LightGBM Model with roberta-encoding

# +
lgb2_models = {}
lgb2_preds = {}

for score in cols:
    lgb2_params = {'objective': 'regression',
                   'metric': 'rmse',
                   'verbosity': 0,
                   'early_stopping_round': 50,
                   'random_state': seed,
                   'device': 'gpu'}
    
    train_set=lgb_trains_roberta[score]
    valid_sets=lgb_vals_roberta[score]

    lgb2_model = lgb.train(
        params=lgb2_params,
        train_set=train_set,
        num_boost_round=1000,
        valid_sets=(train_set, valid_sets),
        callbacks=None,
        verbose_eval=100
    )
    
    lgb2_models[score] = lgb2_model
    lgb2_preds[score] = lgb2_model.predict(X_test_roberta_3d[:,1,:,0])
    
lgb2_pred = pd.DataFrame(lgb2_preds, index=X_test_idx)
lgb2_pred
# -

preds = [keras1_pred, keras2_pred, keras3_pred, keras4_pred, lgb1_pred, lgb2_pred]
pred = pd.DataFrame(np.mean(np.array(preds), axis=0), columns=cols, index=X_test_idx)
pred

pred.to_csv("submission.csv", index=True)
