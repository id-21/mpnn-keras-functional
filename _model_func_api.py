from _model_func import upFunc_GRU, msgFunc_EN, create_mpnn_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
import numpy as np

batch_size = 10
n_node = 4
node_dim = 5
edge_dim = 3
hidden_dim = 6
n_step = 4
d = 7

print("Successfully imported!")
print("batch_size = 10, n_node = 4, node_dim = 5, edge_dim = 3, hidden_dim = 6, n_step = 2, d = 7")

model = create_mpnn_model(n_step, batch_size, n_node, d, edge_dim)
model.summary()