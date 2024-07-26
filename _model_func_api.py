from _model_func import node_featurizer, edge_featurizer, node_preproc, edge_preproc, create_mpnn_model, GRUUpdateLayer

# import tensorflow
# import keras
# from tensorflow.keras.util import plot_model

batch_size = 100
n_node = 29
node_dim = 10
edge_dim = 10
hidden_dim = 50
n_step = 8

print("Successfully imported!")

# model_1 = node_featurizer(29, 50, 'trial')
# model_1.summary()
# model_2 = edge_featurizer(edge_dim, hidden_dim, 'trial2')
# out = model_2(model_1.output)
# model_3 = tf.keras.Model(model_1.input, out, name='trial1+2')
# model_3.summary()

model = node_preproc(batch_size, n_node, node_dim, hidden_dim)
model.summary()

# model2 = edge_preproc(batch_size, n_node, node_dim, hidden_dim)
# model2.summary()

trial = GRUUpdateLayer(batch_size, n_node, hidden_dim)
trial.trial()

# model3 = create_mpnn_model(n_step, batch_size, n_node, hidden_dim)
# model3.summary()

# plot_model(model,to_file='demo.png',show_shapes=True)

# Remember, you must compute the mask and pass it to the model with the input
# mask can be prepared using:
# mask = tf.clip_by_value(tf.reduce_max(inp, -1, keepdims=True), 0, 1)