# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization, Add, Dropout
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from keras.models import Model, load_model
from keras import callbacks, Sequential
from keras import backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay, plot_precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

class Features:
  
  def __init__(self, num, window):

    dictio = {1: ["chr01", "chrI"],
                 2: ["chr02", "chrII"],
                 3: ["chr03", "chrIII"],
                 4: ["chr04", "chrIV"],
                 5: ["chr05", "chrV"],
                 6: ["chr06", "chrVI"],
                 7: ["chr07", "chrVII"],
                 8: ["chr08", "chrVIII"],
                 9: ["chr09", "chrIX"],
                 10: ["chr10", "chrX"],
                 11: ["chr11", "chrXI"],
                 12: ["chr12", "chrXII"],
                 13: ["chr13", "chrXIII"],
                 14: ["chr14", "chrXIV"],
                 15: ["chr15", "chrXV"],
                 16: ["chr16", "chrXVI"]}

    chr_l = []
    chr_n = []
    for i in range(len(num)):
      chr_l.append(dictio[num[i]][0])
      chr_n.append(dictio[num[i]][1])

    self.chr_l = chr_l
    self.chr_n = chr_n
    self.num = num
    self.window = window
    self.featurize()

  def get_sequences(self):

    all_seqs = []
    for i in range(len(self.chr_l)):
      chr = pd.read_csv("data/" + self.chr_l[i] + ".fsa")

      sequence  = ""
      for i in range(len(chr)):
        sequence = sequence + chr.loc[i][0]
      all_seqs.append(sequence)

    return(all_seqs)

  def get_energy(self):
    
    with open("data/ene.json") as f:
      ene = json.load(f)

    ene_chr = np.array(ene)[np.array(self.num) - [1]*len(self.num)]
    for i in range(len(ene_chr)):
      ene_chr[i] = (ene_chr[i]-np.min(ene_chr[i])) / (np.max(ene_chr[i]) - np.min(ene_chr[i]))

    ene_norm = ene_chr

    return(ene_norm)

  def create_tfbs(self, tf, seq):

    tf_seq = [0] * len(seq)
    for i in range(len(tf)):
      s = min(tf['start'][i], tf['end'][i])
      e = max(tf['start'][i], tf['end'][i])
      tf_seq[(s-1):e] = [tf['score'][i]] * ((e + 1) - s)
    
    return(tf_seq)

  def get_tfbs(self, seq):

    all_tfbs = pd.read_csv("data/sites.csv")
    tfbs_l = []
    for i in range(len(self.chr_n)):
      tfbs_t = all_tfbs[all_tfbs.seqid == self.chr_n[i]]
      tfbs_t = tfbs_t.reset_index()
      tfbs_l.append(self.create_tfbs(tfbs_t, seq[i]))

    return(tfbs_l)

  def featurize(self, testing_dataset=None):

    seq = self.get_sequences()
    self.ene_norm = self.get_energy()
    self.tfbs_l = self.get_tfbs(seq)

    # positions will take the center of nfr to get window of descriptor
    tss = pd.read_csv("data/tss.classes.csv")
    tss = tss[(tss['descr'] == "W-open-W") | (tss['descr'] == "W-closed-W")]
    tss = tss[tss['dist'] <= self.window]
    # tts = pd.read_csv("data/tts.classes.csv")
    # tts = tts[(tts['descr'] == "W-open-W") | (tts['descr'] == "W-closed-W")].reset_index()

    tss = tss[tss.seqname.isin(self.chr_n)]

    self.energy =[]
    self.tfbsites = []
    self.labels = []

    for i in range(len(self.chr_n)):
      tss_calls = tss[tss['seqname'] == self.chr_n[i]].reset_index()
      positions_chr = tss_calls['start'] + round((tss_calls['end']  - tss_calls['start'])/2)
      positions_chr = np.array(positions_chr)
      self.nfr_len = np.array(abs(tss_calls['end']  - tss_calls['start']))

      for j in range(len(positions_chr)):
        s = int(positions_chr[j] - (self.window/2)) - 1
        e = int(positions_chr[j] + (self.window/2)) - 1 # DOUBLE CHRCK INDEXING
        self.energy.append(self.ene_norm[i][s:e])
        self.tfbsites.append(np.array(self.tfbs_l[i])[s:e])

      self.labels = np.hstack([self.labels, ["NFR"] * len(positions_chr)])

      # We will get positions of non nfr by adding 200 bases to the end positions

      positions_chr = tss_calls['end']+ 250 
      positions_chr = np.array(positions_chr)

      for j in range(len(positions_chr)): 
        s = int(positions_chr[j] - (self.window/2)) - 1
        e = int(positions_chr[j] + (self.window/2)) - 1
        self.energy.append(self.ene_norm[i][s:e])
        self.tfbsites.append(np.array(self.tfbs_l[i])[s:e])

      self.nfr_len = np.hstack([self.nfr_len, [0] * len(positions_chr)])
      self.labels = np.hstack([self.labels, ["Nuc"] * len(positions_chr)])

    self.features = np.hstack((self.energy, self.tfbsites))
    return(self.energy, self.tfbsites, self.features, self.labels)

class Model:
  
  def __init__(self, dataset):

    self.data = dataset
    self.X = self.data.features
    self.ene_norm = dataset.ene_norm
    self.tfbs_l = dataset.tfbs_l
    labels = self.data.labels

    self.encoder = LabelEncoder()
    self.encoder.fit(labels)
    encoded_Y = self.encoder.transform(labels)
    # convert integers to one hot encoded
    self.y = np_utils.to_categorical(encoded_Y)

    arr = np.arange(len(self.y))
    np.random.shuffle(arr)
    self.y = self.y[arr]
    self.X = self.X[arr]

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20)

    self.model = Sequential()
    self.model.add(Dense(30, activation='relu'))
    self.model.add(Dense(2, activation='sigmoid'))

    self.model.compile(loss='binary_crossentropy',
                       optimizer=SGD(lr = 0.001, momentum = 0.003),
                       metrics=['acc',
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall()])

  def predictor(self, testing_dataset=None):
   # if testing_dataset is not None:
        #self.X_test, labels = testing_dataset.features, testing_dataset.labels
        #encoded_Y = encoder.transform(labels)
        #self.y_test = np_utils.to_categorical(encoded_Y)
   #     self.X_test = testing_dataset
        
    self.history = self.model.fit(self.X_train, self.y_train,
                  batch_size=10,
                  epochs=500,
                  verbose=0,
                  validation_data=(self.X_test, self.y_test))
    
    self.score = self.model.evaluate(self.X_test, self.y_test, verbose=0)

  def plot_learning(self, history):
    history = self.history
    keys = list(history.history.keys())
    # summarize history for metrics
    plt.plot(history.history[keys[1]])
    plt.plot(history.history[keys[5]])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.show()

    plt.plot(history.history[keys[2]])
    plt.plot(history.history[keys[6]])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['precision', 'val_precision'], loc='upper left')
    plt.show()

    plt.plot(history.history[keys[3]])
    plt.plot(history.history[keys[7]])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['recall', 'val_recall'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history[keys[0]])
    plt.plot(history.history[keys[4]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()
