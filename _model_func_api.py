from _model_func import upFunc_GRU, msgFunc_EN, create_mpnn_model
from util import _permutation


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
import numpy as np
import pickle as pkl
import csv, sys
from sklearn.preprocessing import StandardScaler

atom_list=['H','C','N','O','F']
batch_size = 100
n_node = 29
node_dim = 25
edge_dim = 13
hidden_dim = 13
n_step = 4
# Must be greater than 25
d = 30
t = 9
n_hidden = 8
prop_d = 11

print("Successfully imported!")
print(f"batch_size = {batch_size}, n_node = {n_node}, node_dim = {node_dim}")
print(f"edge_dim = {edge_dim}, hidden_dim = {hidden_dim}, n_step = {n_step}, d = {d}")
print(f"t = {t}, n_hidden = {n_hidden}, prop_d = {prop_d}")

model = create_mpnn_model(n_step, batch_size, n_node, d, edge_dim, t, n_hidden, prop_d)
model.summary()

data_path = './QM9_graph_2.pkl'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DP, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
# DP = np.expand_dims(DP, 3)

scaler = StandardScaler()
DY = scaler.fit_transform(DY)

dim_atom = len(atom_list)
dim_y = DY.shape[1]
print(DV.shape, DE.shape, DP.shape, DY.shape)

n_tst = 10000
n_val = 10000
n_trn = 10000

print(':: preprocess data')
np.random.seed(134)

DE = np.concatenate((DE, DP), axis = 3)
print(d-DV.shape[2])

# DV = DV[:3*n_tst]
# DE = DE[:3*n_tst]
# DY = DY[:3*n_tst]

DV = np.pad(DV, [(0, 0), (0, 0), (0, d-DV.shape[2])], mode = 'constant', constant_values = 0)

print(DV.shape, DE.shape, DP.shape, DY.shape)

# exit()

[DV, DE, DP, DY, Dsmi] = _permutation([DV, DE, DP, DY, Dsmi])


x = np.split(DY, dim_y, 1)
DY = x[0]
print(DY.shape)

DV_trn = DV[:n_trn]
DE_trn = DE[:n_trn]
DP_trn = DP[:n_trn]
DY_trn = DY[:n_trn]
    
DV_val = DV[n_trn:n_trn+n_val]
DE_val = DE[n_trn:n_trn+n_val]
DP_val = DP[n_trn:n_trn+n_val]
DY_val = DY[n_trn:n_trn+n_val]

DV_tst = DV[-n_tst:]
DE_tst = DE[-n_tst:]
DP_tst = DP[-n_tst:]
DY_tst = DY[-n_tst:]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = model.fit(
    [DE_trn, DV_trn],
    DY_trn,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)


plot_loss(history)

test_results = {}

test_results['horsepower_model'] = model.evaluate(
    [DE_val, DV_val],
    DY_val, verbose=0)