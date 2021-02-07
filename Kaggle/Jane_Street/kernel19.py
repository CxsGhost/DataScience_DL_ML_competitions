from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, \
    Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices

import kerastuner as kt



physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
            y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]

            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=callbacks)

            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id,
                                 {k: np.mean(val_losses[:, i]) for i, k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


### Loading the training data

TRAINING = 1
USE_FINETUNE = 0
FOLDS = 5
SEED = 68

train = pd.read_csv('jane-street-market-prediction/train.csv')
train = train.query('date > 85').reset_index(drop=True)
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})  # limit memory use
train.fillna(train.mean(), inplace=True)
train = train.query('weight > 0').reset_index(drop=True)
# train['action'] = (train['resp'] > 0).astype('int')
train['action'] = ((train['resp_1'] > 0.00001) & (train['resp_2'] > 0.00001) & (train['resp_3'] > 0.00001) & (
            train['resp_4'] > 0.00001) & (train['resp'] > 0.00001)).astype('int')
features = [c for c in train.columns if 'feature' in c]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X = train[features].values
y = np.stack([(train[c] > 0.000001).astype('int') for c in resp_cols]).T  # Multitarget

f_mean = np.mean(train[features[1:]].values, axis=0)



### Creating the autoencoder.

def create_autoencoder(input_dim, output_dim, noise=0.1):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(input_dim, name='decoded')(encoded)
    x = Dense(128, activation='relu')(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)

    encoder = Model(inputs=i, outputs=encoded)
    autoencoder = Model(inputs=i, outputs=[decoded, x])

    autoencoder.compile(optimizer=Adam(0.005), loss={'decoded': 'mse', 'label_output': 'binary_crossentropy'})
    return autoencoder, encoder



def create_model(hp, input_dim, output_dim, encoder):
    inputs = Input(input_dim)

    x = encoder(inputs)
    x = Concatenate()([x, inputs])  # use both raw and encoded features
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('init_dropout', 0.0, 0.5))(x)

    for i in range(hp.Int('num_layers', 1, 3)):
        x = Dense(hp.Int('num_units_{i}', 64, 256))(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5))(x)
    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(hp.Float('lr', 0.0005, 0.1, default=0.001)),
                  loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing', 0.0, 0.1)),
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model



### Defining and training the autoencoder.

autoencoder, encoder = create_autoencoder(X.shape[-1], y.shape[-1], noise=0.1)
if TRAINING:
    autoencoder.fit(X, (X, y),
                    epochs=1010,
                    batch_size=16384,
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss', patience=45, restore_best_weights=True),
                               ReduceLROnPlateau(monitor='val_loss', patience=12, factor=0.5),
                               ModelCheckpoint('./autoencoder.hdf5', verbose=1, monitor='val_loss', save_best_only=True,
                                               save_weights_only=True)])
    autoencoder.load_weights('./autoencoder.hdf5')
    encoder.save_weights('./encoder.hdf5')
else:
    encoder.load_weights('janest-weight/vae005/encoder.hdf5')
encoder.trainable = False

### Running CV

model_fn = lambda hp: create_model(hp, X.shape[-1], y.shape[-1], encoder)

tuner = CVTuner(
    hypermodel=model_fn,
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('val_auc', direction='max'),
        num_initial_points=4,
        max_trials=50))

FOLDS = 5
SEED = 27

if TRAINING:
    gkf = PurgedGroupTimeSeriesSplit(n_splits=FOLDS, group_gap=20)
    splits = list(gkf.split(y, groups=train['date'].values))
    tuner.search((X,), (y,), splits=splits, batch_size=16384, epochs=300,
                 callbacks=[EarlyStopping('val_auc', mode='max', patience=3)])
    hp = tuner.get_best_hyperparameters(1)[0]
    pd.to_pickle(hp, f'./best_hp_{SEED}.pkl')
    for fold, (train_indices, test_indices) in enumerate(splits):
        model = model_fn(hp)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=16384,
                  callbacks=[EarlyStopping('val_auc', mode='max', patience=10, restore_best_weights=True)])
        model.save_weights(f'./model_{SEED}_{fold}.hdf5')
        model.compile(Adam(hp.get('lr') / 100), loss='binary_crossentropy')
        model.fit(X_test, y_test, epochs=6, batch_size=16384)
        model.save_weights(f'./model_{SEED}_{fold}_finetune.hdf5')
    tuner.results_summary()
else:
    models = []
    hp = pd.read_pickle(f'../input/janest-weight/vae005/best_hp_{SEED}.pkl')
    for f in range(FOLDS):
        model = model_fn(hp)
        if USE_FINETUNE:
            model.load_weights(f'../input/janest-weight/vae005/model_{SEED}_{f}_finetune.hdf5')
        else:
            model.load_weights(f'../input/janest-weight/vae005/model_{SEED}_{f}.hdf5')
        models.append(model)

## Submission

if not TRAINING:
    f = np.median
    models = models[-2:]
    import janestreet

    env = janestreet.make_env()
    th = 0.5
    for (test_df, pred_df) in tqdm(env.iter_test()):
        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, features].values
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
            pred = np.mean([model(x_tt, training=False).numpy() for model in models], axis=0)
            pred = f(pred)
            pred_df.action = np.where(pred >= th, 1, 0).astype(int)
        else:
            pred_df.action = 0
        env.predict(pred_df)