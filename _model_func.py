# PENDING TASKS
# 1. Figure out how to apply masks using Keras Layers Embedding OR fix Keras.Multiply used to apply mask
# https://www.tensorflow.org/guide/keras/understanding_masking_and_padding

# Must set these before importing tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.autograph.set_verbosity(0)

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Reshape, Lambda, Layer, multiply
from tensorflow.keras.models import Sequential

# Dummy featurizer written to test Keras Functional API
# Used this backbone in node_preproc
# Not necessary for the code, not used in any other part
def node_featurizer(node_dim, hidden_dim, name):
    # model = Sequential([
    #     Input(shape=(node_dim,)),
    #     Dense(hidden_dim*5, activation = 'relu'),
    #     Dense(hidden_dim*5, activation = 'relu'),
    #     Dense(hidden_dim*5, activation = 'relu'),
    #     Dense(hidden_dim, activation = 'tanh'),
    # ])
    inp = Input(shape=[node_dim, ])
    x = Dense(hidden_dim*5, activation = 'relu')(inp)
    x = Dense(hidden_dim*5, activation = 'relu')(x)
    x = Dense(hidden_dim*5, activation = 'relu')(x)
    out = Dense(hidden_dim, activation = 'tanh')(x)
    return tf.keras.Model(inp, out, name=name)

#
def edge_featurizer(edge_dim, hidden_dim, name):
    # model = Sequential([
    #     Input(shape=[edge_dim,]),                # input layer
    #     Dense(hidden_dim*5, activation='relu'),
    #     Dense(hidden_dim*5, activation='relu'),
    #     Dense(hidden_dim*5, activation='relu'),
    #     Dense(hidden_dim*hidden_dim)             # output layer
    # ])
    inp = Input(shape=[edge_dim, ])
    x = Dense(hidden_dim*5, activation = 'relu')(inp)
    x = Dense(hidden_dim*5, activation = 'relu')(x)
    x = Dense(hidden_dim*5, activation = 'relu')(x)
    out = Dense(hidden_dim*hidden_dim)(x)
    return tf.keras.Model(inp, out, name=name)

def node_preproc(batch_size, n_node, node_dim, hidden_dim):
    def apply_mask(x):
        node_feat, mask = x
        return node_feat*mask

    node_feat_inp = Input(shape=[batch_size, n_node, node_dim], name='node_feat')
    # If using layers.Multiply, mask_inp and node_feat_inp must be of same dimension
    mask_inp = Input(shape=[batch_size, n_node, hidden_dim], name='mask')
    re_inp = Reshape((batch_size*n_node, node_dim))(node_feat_inp)
    x1 = Dense(hidden_dim*5, activation = 'relu')(re_inp)
    x2 = Dense(hidden_dim*5, activation = 'relu')(x1)
    x3 = Dense(hidden_dim*5, activation = 'relu')(x2)
    x4 = Dense(hidden_dim)(x3)
    x5 = Reshape((batch_size, n_node, hidden_dim))(x4)
    out_shape = x5.shape
    # print(out_shape, mask_inp.shape)
    masked_out = multiply([x5, mask_inp])
    # masked_out = Lambda(apply_mask, output_shape = out_shape)([x5, mask_inp])
    re_out = Reshape((batch_size, n_node, hidden_dim))(masked_out)
    return tf.keras.Model(inputs=[node_feat_inp, mask_inp], outputs=re_out, name='Node Preprocessing')

def edge_preproc(batch_size, n_node, edge_dim, hidden_dim):
    def remove_self_loops(edge_feat):
        return edge_feat * tf.reshape(1-tf.eye(n_node), [1, n_node, n_node, 1, 1])
    def remove_invalid_edges(x):
        edge_feat, mask = x
        edge_feat = edge_feat * tf.reshape(mask, [batch_size, n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, n_node, 1, 1])
        return edge_feat    
    edge_feat_inp = Input(shape=[batch_size, n_node, n_node, edge_dim])
    mask_inp = Input(shape=[batch_size, n_node, 1], name='mask')
    re_inp = Reshape((batch_size*n_node*n_node, edge_dim))(edge_feat_inp)
    x1 = Dense(hidden_dim*5, activation = 'relu')(re_inp)
    x2 = Dense(hidden_dim*5, activation = 'relu')(x1)
    x3 = Dense(hidden_dim*5, activation = 'relu')(x2)
    x4 = Dense(hidden_dim*hidden_dim)(x3)
    x5 = Reshape((batch_size, n_node, n_node, hidden_dim, hidden_dim))(x4)
    x6 = Lambda(remove_self_loops, output_shape = x5.shape)(x5)
    x7 = Lambda(remove_invalid_edges, output_shape = x5.shape)([x6, mask_inp])
    return tf.keras.Model(inputs=[edge_feat_inp, mask_inp], outputs=x7, name='Edge Preprocessing')

# def _edge_nn(inp, mask, batch_size, n_node, edge_dim, hidden_dim):
#     inp = tf.reshape(inp, [batch_size * n_node * n_node, edge_dim])
#     edge_model = edge_featurizer(edge_dim, hidden_dim, 'edge_pp_nn')
#     out = edge_model(inp)
#     out = tf.reshape(out, [batch_size, n_node, n_node, hidden_dim, hidden_dim])
#     # Remove self Loops
#     out = out * tf.reshape(1-tf.eye(n_node), [1, n_node, n_node, 1, 1])
#     # Remove invalid edges using mask
#     out = out * tf.reshape(mask, [batch_size, n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, n_node, 1, 1])
#     return out

class _msg_nn(Layer):
    def __init__(self, batch_size, n_node, hidden_dim, **kwargs):
        super(_msg_nn, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n_node = n_node
        self.hidden_dim = hidden_dim
    
    def call(self, inputs):
        edge_wgt, node_hidden = inputs
        wgt = tf.reshape(edge_wgt, [self.batch_size*self.n_node, self.n_node*self.hidden_dim, self.hidden_dim])
        node = tf.reshape(node_hidden, [self.batch_size*self.n_node, self.hidden_dim, 1])
        msg = tf.matmul(wgt, node)
        msg = tf.reshape(msg, [self.batch_size, self.n_node, self.n_node, self.hidden_dim])
        msg = tf.transpose(msg, perm=[0, 2, 3, 1])
        msg = tf.reduce_mean(msg, 3)
        return msg

class GRUUpdateLayer(Layer):
    def __init__(self, batch_size, n_node, hidden_dim, **kwargs):
        super(GRUUpdateLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n = n_node
        self.d = hidden_dim
        # Comment this out when hidden_dim code has been replaced
        self.hidden_dim = hidden_dim
        self.n_node = n_node

        self.gru = GRU(self.d * self.n, return_sequences=True, return_state=True)
    
    # Rewrite call keeping in mind d instead of hidden_dim    
    def call(self, inputs):
        msg, node = inputs
        msg = tf.reshape(msg, [self.batch_size, 1, self.d*self.n])
        node = tf.reshape(node, [self.batch_size, self.d*self.n])
        print("msg: ", msg.shape)
        print("node: ", node.shape)
        node_next, state = self.gru(msg, initial_state = node)
        # node_next = tf.reshape(node_next, [self.batch_size, self.n_node, self.hidden_dim])
        return node_next, state
    
    def trial(self):
        batch_s = 4
        dim = 3
        seq = 1

        # According to this link, input should be of shape: [batch_size, seq_len, input_dim]
        # https://discuss.pytorch.org/t/gru-for-multi-dimensional-input/156682
        inputs = np.random.random((batch_s, seq, dim))
        initial = np.ones((batch_s, dim))

        inp_shape = np.shape(inputs)
        print("Input Shape: ", np.shape(inputs))
        print("Initial Shape: ", np.shape(initial))
        # print("Input Shape Modified: ", np.shape(inputs[0]))
        gru = GRU(dim, return_sequences=True, return_state = True)
        seq, state = gru(inputs, initial_state = initial)
        print(initial)
        print(seq)
        print(state)
        return

    def trial2(self):
        batch_s = 4
        dim = 3
        seq = 5

        inputs = np.random.random((batch_s, seq, dim))
        initial = np.ones((batch_s, dim))
        print(inputs.shape)

        gru = GRU(dim, return_sequences=True, return_state=True)
        state = initial
        for i in range(seq):
            inp = inputs[:, i:i+1, :]
            # print(inp.shape)
            seq, n_state = gru(inp, initial_state = state)
            print(seq.shape, n_state.shape)
            state = n_state

        return


def test(n_step, batch_size, n_node, hidden_dim):
    pass

def create_mpnn_model(n_step, batch_size, n_node, hidden_dim):
    edge_wt_input = Input(shape=(n_node, n_node, hidden_dim, hidden_dim), batch_size=batch_size, name='edge_wt')
    node_hidden_input = Input(shape=(n_node, hidden_dim), batch_size=batch_size, name='node_hidden')
    # mask_input = Input(shape=(n_node, hidden_dim), batch_size=batch_size, name='mask')

    node_hidden = node_hidden_input
    nhs = []

    gru = GRUUpdateLayer(batch_size, n_node, hidden_dim)
    for _ in range(n_step):
        msg = _msg_nn(batch_size, n_node, hidden_dim)([edge_wt_input, node_hidden])
        node_hidden, state = gru([msg, node_hidden])
        nhs.append(node_hidden)

    out = tf.concat(nhs, axis = 2)
    model = tf.Model(inputs=[edge_wt_input, node_hidden_input], outputs=out)
    return model