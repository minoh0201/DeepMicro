# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# importing keras
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle

#fix np.random.seed for reproducibility in numpy processing
np.random.seed(7)

class DeepMicrobiome(object):
    def __init__(self, data, seed, data_dir):
        self.t_start = time.time()
        self.filename = str(data)
        self.data = self.filename.split('.')[0]
        self.seed = seed
        self.data_dir = data_dir
        self.prefix = ''
        self.representation_only = False

    def loadData(self, feature_string, label_string, label_dict, dtype=None):
        # read file
        filename = self.data_dir + "data/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T

        # get class labels
        Y = raw.loc[label_string] #'disease'
        Y = Y.replace(label_dict)

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
        self.printDataShapes()

    def loadCustomData(self, dtype=None):
        # read file
        filename = self.data_dir + "data/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # load data
        self.X_train = raw.values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1,self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)
        self.printDataShapes(train_only=True)

    def loadCustomDataWithLabels(self, label_data, dtype=None):
        # read file
        filename = self.data_dir + "data/" + self.filename
        label_filename = self.data_dir + "data/" + label_data
        if os.path.isfile(filename) and os.path.isfile(label_filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
            label = pd.read_csv(label_filename, sep=',', index_col=False, header=None)
        else:
            if not os.path.isfile(filename):
                print("FileNotFoundError: File {} does not exist".format(filename))
            if not os.path.isfile(label_filename):
                print("FileNotFoundError: File {} does not exist".format(label_filename))
            exit()

        # label data validity check
        if not label.values.shape[1] > 1:
            label_flatten = label.values.reshape((label.values.shape[0]))
        else:
            print('FileSpecificationError: The label file contains more than 1 column.')
            exit()

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(raw.values.astype(dtype),
                                                                                label_flatten.astype('int'), test_size=0.2,
                                                                                random_state=self.seed,
                                                                                stratify=label_flatten)
        self.printDataShapes()


    #Principal Component Analysis
    def pca(self, ratio=0.99):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'PCA_'

        # PCA
        pca = PCA()
        pca.fit(self.X_train)
        n_comp = 0
        ratio_sum = 0.0

        for comp in pca.explained_variance_ratio_:
            ratio_sum += comp
            n_comp += 1
            if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
                break

        pca = PCA(n_components=n_comp)
        pca.fit(self.X_train)

        X_train = pca.transform(self.X_train)
        X_test = pca.transform(self.X_test)

        # applying the eigenvectors to the whole training and the test set.
        self.X_train = X_train
        self.X_test = X_test
        self.printDataShapes()

    #Gausian Random Projection
    def rp(self):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'RandP_'
        # GRP
        rf = GaussianRandomProjection(eps=0.5)
        rf.fit(self.X_train)

        # applying GRP to the whole training and the test set.
        self.X_train = rf.transform(self.X_train)
        self.X_test = rf.transform(self.X_test)
        self.printDataShapes()

    #Shallow Autoencoder & Deep Autoencoder
    def ae(self, dims = [50], epochs= 2000, batch_size=100, verbose=2, loss='mean_squared_error', latent_act=False, output_act=False, act='relu', patience=20, val_rate=0.2, no_trn=False):

        # manipulating an experiment identifier in the output file
        if patience != 20:
            self.prefix += 'p' + str(patience) + '_'
        if len(dims) == 1:
            self.prefix += 'AE'
        else:
            self.prefix += 'DAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if latent_act:
            self.prefix += 't'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix = self.prefix + 's'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train, test_size=val_rate, random_state=self.seed, stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.autoencoder, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
        self.autoencoder.summary()

        if no_trn:
            return

        # compile model
        self.autoencoder.compile(optimizer='adam', loss=loss)

        # fit model
        self.history = self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                             verbose=verbose, validation_data=(X_inner_test, X_inner_test))
        # save loss progress
        self.saveLossProgress()

        # load best model
        self.autoencoder = load_model(modelName)
        layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
        self.encoder = Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)

    # Variational Autoencoder
    def vae(self, dims = [10], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, no_trn=False):

        # manipulating an experiment identifier in the output file
        if patience != 25:
            self.prefix += 'p' + str(patience) + '_'
        if warmup:
            self.prefix += 'w' + str(warmup_rate) + '_'
        self.prefix += 'VAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        if beta != 1:
            self.prefix += 'B' + str(beta)
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]

        # warm-up callback
        warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])  # , print(epoch), print(K.get_value(beta))])

        # warm-up implementation
        def warm_up(epoch):
            val = epoch * warmup_rate
            if val <= 1.0:
                K.set_value(beta, val)
        # add warm-up callback if requested
        if warmup:
            beta = K.variable(value=0.0)
            callbacks.append(warm_up_cb)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create vae model
        self.vae, self.encoder, self.decoder = DNN_models.variational_AE(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)
        self.vae.summary()

        if no_trn:
            return

        # fit
        self.history = self.vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))

        # save loss progress
        self.saveLossProgress()

        # load best model
        self.vae.load_weights(modelName)
        self.encoder = self.vae.layers[1]

        # applying the learned encoder into the whole training and the test set.
        _, _, self.X_train = self.encoder.predict(self.X_train)
        _, _, self.X_test = self.encoder.predict(self.X_test)

    # Convolutional Autoencoder
    def cae(self, dims = [32], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, val_rate=0.2, rf_rate = 0.1, st_rate = 0.25, no_trn=False):

        # manipulating an experiment identifier in the output file
        self.prefix += 'CAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]


        # fill out blank
        onesideDim = int(math.sqrt(self.X_train.shape[1])) + 1
        enlargedDim = onesideDim ** 2
        self.X_train = np.column_stack((self.X_train, np.zeros((self.X_train.shape[0], enlargedDim - self.X_train.shape[1]))))
        self.X_test = np.column_stack((self.X_test, np.zeros((self.X_test.shape[0], enlargedDim - self.X_test.shape[1]))))

        # reshape
        self.X_train = np.reshape(self.X_train, (len(self.X_train), onesideDim, onesideDim, 1))
        self.X_test = np.reshape(self.X_test, (len(self.X_test), onesideDim, onesideDim, 1))
        self.printDataShapes()

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, (onesideDim, onesideDim, 1))

        # create cae model
        self.cae, self.encoder = DNN_models.conv_autoencoder(dims, act=act, output_act=output_act, rf_rate = rf_rate, st_rate = st_rate)
        self.cae.summary()
        if no_trn:
            return

        # compile
        self.cae.compile(optimizer='adam', loss=loss)

        # fit
        self.history = self.cae.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, X_inner_test, None))

        # save loss progress
        self.saveLossProgress()

        # load best model
        self.cae.load_weights(modelName)
        if len(self.cae.layers) % 2 == 0:
            layer_idx = int((len(self.cae.layers) - 2) / 2)
        else:
            layer_idx = int((len(self.cae.layers) - 1) / 2)
        self.encoder = Model(self.cae.layers[0].input, self.cae.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)
        self.printDataShapes()

    # Classification
    def classification(self, hyper_parameters, method='svm', cv=5, scoring='roc_auc', n_jobs=1, cache_size=10000):
        clf_start_time = time.time()

        print("# Tuning hyper-parameters")
        print(self.X_train.shape, self.y_train.shape)

        # Support Vector Machine
        if method == 'svm':
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100, )
            clf.fit(self.X_train, self.y_train)

        # Random Forest
        if method == 'rf':
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train)

        # Multi-layer Perceptron
        if method == 'mlp':
            model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=self.X_train.shape[1], verbose=0, )
            clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train, batch_size=32)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

        # Evaluate performance of the best model on test set
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        y_prob = clf.predict_proba(self.X_test)

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        metrics = [round(roc_auc_score(y_true, y_prob[:, 1]), 4),
                   round(accuracy_score(y_true, y_pred), 4),
                   round(recall_score(y_true, y_pred), 4),
                   round(precision_score(y_true, y_pred), 4),
                   round(f1_score(y_true, y_pred), 4), ]

        # time stamp
        metrics.append(str(datetime.datetime.now()))

        # running time
        metrics.append(round( (time.time() - self.t_start), 2))

        # classification time
        metrics.append(round( (time.time() - clf_start_time), 2))

        # best hyper-parameter append
        metrics.append(str(clf.best_params_))

        # Write performance metrics as a file
        res = pd.DataFrame([metrics], index=[self.prefix + method])
        with open(self.data_dir + "results/" + self.data + "_result.txt", 'a') as f:
            res.to_csv(f, header=None)

        print('Accuracy metrics')
        print('AUC, ACC, Recall, Precision, F1_score, time-end, runtime(sec), classfication time(sec), best hyper-parameter')
        print(metrics)

    def printDataShapes(self, train_only=False):
        print("X_train.shape: ", self.X_train.shape)
        if not train_only:
            print("y_train.shape: ", self.y_train.shape)
            print("X_test.shape: ", self.X_test.shape)
            print("y_test.shape: ", self.y_test.shape)

    # ploting loss progress over epochs
    def saveLossProgress(self):
        #print(self.history.history.keys())
        #print(type(self.history.history['loss']))
        #print(min(self.history.history['loss']))

        loss_collector, loss_max_atTheEnd = self.saveLossProgress_ylim()

        # save loss progress - train and val loss only
        figureName = self.prefix + self.data + '_' + str(self.seed)
        plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'],
                   loc='upper right')
        plt.savefig(self.data_dir + "results/" + figureName + '.png')
        plt.close()

        if 'recon_loss' in self.history.history:
            figureName = self.prefix + self.data + '_' + str(self.seed) + '_detailed'
            plt.ylim(min(loss_collector) * 0.9, loss_max_atTheEnd * 2.0)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.plot(self.history.history['recon_loss'])
            plt.plot(self.history.history['val_recon_loss'])
            plt.plot(self.history.history['kl_loss'])
            plt.plot(self.history.history['val_kl_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss', 'recon_loss', 'val recon_loss', 'kl_loss', 'val kl_loss'], loc='upper right')
            plt.savefig(self.data_dir + "results/" + figureName + '.png')
            plt.close()

    # supporting loss plot
    def saveLossProgress_ylim(self):
        loss_collector = []
        loss_max_atTheEnd = 0.0
        for hist in self.history.history:
            current = self.history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_atTheEnd:
                loss_max_atTheEnd = current[-1]
        return loss_collector, loss_max_atTheEnd

if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()

    # load data
    load_data = parser.add_argument_group('Loading data')
    load_data.add_argument("-d", "--data", help="prefix of dataset to open (e.g. abundance_Cirrhosis)", type=str,
                        choices=["abundance_Cirrhosis", "abundance_Colorectal", "abundance_IBD",
                                 "abundance_Obesity", "abundance_T2D", "abundance_WT2D",
                                 "marker_Cirrhosis", "marker_Colorectal", "marker_IBD",
                                 "marker_Obesity", "marker_T2D", "marker_WT2D",
                                 ])
    load_data.add_argument("-cd", "--custom_data", help="filename for custom input data under the 'data' folder", type=str,)
    load_data.add_argument("-cl", "--custom_data_labels", help="filename for custom input labels under the 'data' folder", type=str,)
    load_data.add_argument("-p", "--data_dir", help="custom path for both '/data' and '/results' folders", default="")
    load_data.add_argument("-dt", "--dataType", help="Specify data type for numerical values (float16, float32, float64)",
                        default="float64", type=str, choices=["float16", "float32", "float64"])
    dtypeDict = {"float16": np.float16, "float32": np.float32, "float64": np.float64}

    # experiment design
    exp_design = parser.add_argument_group('Experiment design')
    exp_design.add_argument("-s", "--seed", help="random seed for train and test split", type=int, default=0)
    exp_design.add_argument("-r", "--repeat", help="repeat experiment x times by changing random seed for splitting data",
                        default=5, type=int)

    # classification
    classification = parser.add_argument_group('Classification')
    classification.add_argument("-f", "--numFolds", help="The number of folds for cross-validation in the tranining set",
                        default=5, type=int)
    classification.add_argument("-m", "--method", help="classifier(s) to use", type=str, default="all",
                        choices=["all", "svm", "rf", "mlp", "svm_rf"])
    classification.add_argument("-sc", "--svm_cache", help="cache size for svm run", type=int, default=1000)
    classification.add_argument("-t", "--numJobs",
                        help="The number of jobs used in parallel GridSearch. (-1: utilize all possible cores; -2: utilize all possible cores except one.)",
                        default=-2, type=int)
    parser.add_argument("--scoring", help="Metrics used to optimize method", type=str, default='roc_auc',
                        choices=['roc_auc', 'accuracy', 'f1', 'recall', 'precision'])

    # representation learning & dimensionality reduction algorithms
    rl = parser.add_argument_group('Representation learning')
    rl.add_argument("--pca", help="run PCA", action='store_true')
    rl.add_argument("--rp", help="run Random Projection", action='store_true')
    rl.add_argument("--ae", help="run Autoencoder or Deep Autoencoder", action='store_true')
    rl.add_argument("--vae", help="run Variational Autoencoder", action='store_true')
    rl.add_argument("--cae", help="run Convolutional Autoencoder", action='store_true')
    rl.add_argument("--save_rep", help="write the learned representation of the training set as a file", action='store_true')

    # detailed options for representation learning
    ## common options
    common = parser.add_argument_group('Common options for representation learning (SAE,DAE,VAE,CAE)')
    common.add_argument("--aeloss", help="set autoencoder reconstruction loss function", type=str,
                        choices=['mse', 'binary_crossentropy'], default='mse')
    common.add_argument("--ae_oact", help="output layer sigmoid activation function on/off", action='store_true')
    common.add_argument("-a", "--act", help="activation function for hidden layers", type=str, default='relu',
                        choices=['relu', 'sigmoid'])
    common.add_argument("-dm", "--dims",
                        help="Comma-separated dimensions for deep representation learning e.g. (-dm 50,30,20)",
                        type=str, default='50')
    common.add_argument("-e", "--max_epochs", help="Maximum epochs when training autoencoder", type=int, default=2000)
    common.add_argument("-pt", "--patience",
                        help="The number of epochs which can be executed without the improvement in validation loss, right after the last improvement.",
                        type=int, default=20)

    ## AE & DAE only
    AE = parser.add_argument_group('SAE & DAE-specific arguments')
    AE.add_argument("--ae_lact", help="latent layer activation function on/off", action='store_true')

    ## VAE only
    VAE = parser.add_argument_group('VAE-specific arguments')
    VAE.add_argument("--vae_beta", help="weight of KL term", type=float, default=1.0)
    VAE.add_argument("--vae_warmup", help="turn on warm up", action='store_true')
    VAE.add_argument("--vae_warmup_rate", help="warm-up rate which will be multiplied by current epoch to calculate current beta", default=0.01, type=float)

    ## CAE only
    CAE = parser.add_argument_group('CAE-specific arguments')
    CAE.add_argument("--rf_rate", help="What percentage of input size will be the receptive field (kernel) size? [0,1]", type=float, default=0.1)
    CAE.add_argument("--st_rate", help="What percentage of receptive field (kernel) size will be the stride size? [0,1]", type=float, default=0.25)

    # other options
    others = parser.add_argument_group('other optional arguments')
    others.add_argument("--no_trn", help="stop before learning representation to see specified autoencoder structure", action='store_true')
    others.add_argument("--no_clf", help="skip classification tasks", action='store_true')


    args = parser.parse_args()
    print(args)

    # set labels for diseases and controls
    label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
    }

    # hyper-parameter grids for classifiers
    rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']
                            }, ]
    #svm_hyper_parameters_pasolli = [{'C': [2 ** s for s in range(-5, 16, 2)], 'kernel': ['linear']},
    #                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
    #                         'kernel': ['rbf']}]
    svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf']}]
    mlp_hyper_parameters = [{'numHiddenLayers': [1, 2, 3],
                             'epochs': [30, 50, 100, 200, 300],
                             'numUnits': [10, 30, 50, 100],
                             'dropout_rate': [0.1, 0.3],
                             },]


    # run exp function
    def run_exp(seed):

        # create an object and load data
        ## no argument founded
        if args.data == None and args.custom_data == None:
            print("[Error] Please specify an input file. (use -h option for help)")
            exit()
        ## provided data
        elif args.data != None:
            dm = DeepMicrobiome(data=args.data + '.txt', seed=seed, data_dir=args.data_dir)

            ## specify feature string
            feature_string = ''
            data_string = str(args.data)
            if data_string.split('_')[0] == 'abundance':
                feature_string = "k__"
            if data_string.split('_')[0] == 'marker':
                feature_string = "gi|"

            ## load data into the object
            dm.loadData(feature_string=feature_string, label_string='disease', label_dict=label_dict,
                        dtype=dtypeDict[args.dataType])

        ## user data
        elif args.custom_data != None:

            ### without labels - only conducting representation learning
            if args.custom_data_labels == None:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir)
                dm.loadCustomData(dtype=dtypeDict[args.dataType])

            ### with labels - conducting representation learning + classification
            else:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir)
                dm.loadCustomDataWithLabels(label_data=args.custom_data_labels, dtype=dtypeDict[args.dataType])

        else:
            exit()

        numRLrequired = args.pca + args.ae + args.rp + args.vae + args.cae

        if numRLrequired > 1:
            raise ValueError('No multiple dimensionality Reduction')

        # time check after data has been loaded
        dm.t_start = time.time()

        # Representation learning (Dimensionality reduction)
        if args.pca:
            dm.pca()
        if args.ae:
            dm.ae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss,
                  latent_act=args.ae_lact, output_act=args.ae_oact, patience=args.patience, no_trn=args.no_trn)
        if args.vae:
            dm.vae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience= 25 if args.patience==20 else args.patience, beta=args.vae_beta, warmup=args.vae_warmup, warmup_rate=args.vae_warmup_rate, no_trn=args.no_trn)
        if args.cae:
            dm.cae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience=args.patience, rf_rate = args.rf_rate, st_rate = args.st_rate, no_trn=args.no_trn)
        if args.rp:
            dm.rp()

        # write the learned representation of the training set as a file
        if args.save_rep:
            if numRLrequired == 1:
                rep_file = dm.data_dir + "results/" + dm.prefix + dm.data + "_rep.csv"
                pd.DataFrame(dm.X_train).to_csv(rep_file, header=None, index=None)
                print("The learned representation of the training set has been saved in '{}'".format(rep_file))
            else:
                print("Warning: Command option '--save_rep' is not applied as no representation learning or dimensionality reduction has been conducted.")

        # Classification
        if args.no_clf or (args.data == None and args.custom_data_labels == None):
            print("Classification task has been skipped.")
        else:
            # turn off GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            importlib.reload(keras)

            # training classification models
            if args.method == "svm":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
            elif args.method == "rf":
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "mlp":
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "svm_rf":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            else:
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)



    # run experiments
    try:
        if args.repeat > 1:
            for i in range(args.repeat):
                run_exp(i)
        else:
            run_exp(args.seed)

    except OSError as error:
        exception_handle.log_exception(error)
