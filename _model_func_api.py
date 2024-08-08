from _model_func import upFunc_GRU, msgFunc_EN, create_mpnn_model


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
# import keras
# from tensorflow.keras.util import plot_model
import numpy as np

batch_size = 10
n_node = 4
node_dim = 5
edge_dim = 3
hidden_dim = 6
n_step = 2
d = 7

print("Successfully imported!")

trial_msgFunc = msgFunc_EN(batch_size, n_node, edge_dim, d)

trial_adjM = np.random.random((batch_size, n_node, n_node, edge_dim))
a =trial_msgFunc.process(trial_adjM)
# print("trial_adjM: ", trial_adjM.shape)
# print("Edge vectorised: ", a[0][0][0][:].shape)

u = 3
w = 2
trial_h = np.random.random((batch_size, n_node, d))
msg = trial_msgFunc.call(trial_h)
# print("msg shape: ", msg.shape)
# print(msg)

gru = upFunc_GRU(batch_size, n_node, d)
n_next, state = gru([msg, trial_h])

# print("GRU Operation done")
# print("n_next: ", n_next.shape)
# print("state: ", state.shape)
state = tf.reshape(state, [batch_size, n_node, d])
# print(trial_h - state)

# plot_model(model,to_file='demo.png',show_shapes=True)

# Remember, you must compute the mask and pass it to the model with the input
# mask can be prepared using:
# mask = tf.clip_by_value(tf.reduce_max(inp, -1, keepdims=True), 0, 1)

model = create_mpnn_model(n_step, batch_size, n_node, d, edge_dim)

