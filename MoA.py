# https://www.kaggle.com/c/lish-moa/overview - Competition overview
# Forked from https://www.kaggle.com/simakov/keras-multilabel-neural-network-v1-2
# 23814 training observations, 874 features, predicting 206 mechanisms of action. Multilabel problem (multi class but not mutually exclusive).
# score: 0.01648, 1726/4384 teams

import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa
# from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
import time

tic = time.perf_counter()


# %% [code]
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')


def label_smoothing(y_true,y_pred):
    
     return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=0.001)

def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0, 1]))
    return np.mean(metrics)

def preprocess(df):  # Replaces categories control/drug, dose time with 0 and 1
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df


train = preprocess(train_features)
test = preprocess(test_features)

del train_targets['sig_id']

train_targets = train_targets.loc[train['cp_type'] == 0].reset_index(drop=True)  # Drops control rows from train_targets
train = train.loc[train['cp_type'] == 0].reset_index(drop=True)  # Drops control rows from train_features

def create_model(num_columns):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(num_columns),
        tf.keras.layers.Dropout(0.4),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="relu")),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))
    ])
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss=label_smoothing,#'binary_crossentropy'
                  )
    return model

top_feats = [x+1 for x in range(874)]

N_STARTS = 3
tf.random.set_seed(42)

res = train_targets.copy()
ss.loc[:, train_targets.columns] = 0  # Submission format file, reset all to 0
res.loc[:, train_targets.columns] = 0

for seed in range(N_STARTS):
    for n, (tr, te) in enumerate(
            MultilabelStratifiedKFold(n_splits=10, random_state=seed, shuffle=True).split(train_targets, train_targets)):
        print(f'Fold {n}')

        model = create_model(len(top_feats))
        checkpoint_path = f'repeat{seed}_Fold{n}.hdf5'
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, #min_lr=0.001,
                                           mode='min')
        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min')
        model.fit(train.values[tr][:, top_feats],
                  train_targets.values[tr],
                  validation_data=(train.values[te][:, top_feats], train_targets.values[te]),
                  epochs=40, batch_size=128,  # debug epochsfl
                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2
                  )

        model.load_weights(checkpoint_path) # Debug downgraded hdf5 package to version 2.1 from 3.0
        test_predict = model.predict(test.values[:, top_feats])
        val_predict = model.predict(train.values[te][:, top_feats])

        ss.loc[:, train_targets.columns] += test_predict
        res.loc[te, train_targets.columns] += val_predict
        print('')

ss.loc[:, train_targets.columns] /= ((n + 1) * N_STARTS)
res.loc[:, train_targets.columns] /= N_STARTS

print(f'OOF Metric: {metric(train_targets, res)}')

ss.loc[test['cp_type'] == 1, train_targets.columns] = 0 

ss.to_csv('submission.csv', index=False)

toc = time.perf_counter()

print(f"Finished in {toc - tic:0.4f} seconds")
