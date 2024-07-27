from _model_func import upFunc_GRU, msgFunc_EN

# import tensorflow
# import keras
# from tensorflow.keras.util import plot_model
import numpy as np

batch_size = 100
n_node = 29
node_dim = 10
edge_dim = 10
hidden_dim = 50
n_step = 8
d = 13

print("Successfully imported!")

trial_msgFunc = msgFunc_EN(batch_size, n_node, edge_dim, d)

trial_adjM = np.random.random((batch_size, n_node, n_node, edge_dim))
a, b =trial_msgFunc.process(trial_adjM)
print(trial_adjM.shape)
print(a[0][0][0][:].shape)

u = 3
w = 4
trial_h = np.random.random((d, n_node))
# trial_msgFunc(trial_h, u, w)

# plot_model(model,to_file='demo.png',show_shapes=True)

# Remember, you must compute the mask and pass it to the model with the input
# mask can be prepared using:
# mask = tf.clip_by_value(tf.reduce_max(inp, -1, keepdims=True), 0, 1)



