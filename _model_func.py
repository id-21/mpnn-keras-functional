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


def remove_self_loops(edge_feat):
    return edge_feat * tf.reshape(1-tf.eye(n_node), [1, n_node, n_node, 1, 1])
def remove_invalid_edges(x):
    edge_feat, mask = x
    edge_feat = edge_feat * tf.reshape(mask, [batch_size, n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, n_node, 1, 1])
    return edge_feat  

# x6 = Lambda(remove_self_loops, output_shape = x5.shape)(x5)
# x7 = Lambda(remove_invalid_edges, output_shape = x5.shape)([x6, mask_inp])
   

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

# According to this link, input should be of shape: [batch_size, seq_len, input_dim]
# https://discuss.pytorch.org/t/gru-for-multi-dimensional-input/156682
class upFunc_GRU(Layer):
    def __init__(self, batch_size, n_node, hidden_dim, **kwargs):
        super(upFunc_GRU, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n = n_node
        self.d = hidden_dim
        # Comment this out when hidden_dim code has been replaced
        self.hidden_dim = hidden_dim
        self.n_node = n_node

        self.gru = GRU(self.d * self.n, return_sequences=True, return_state=True)
    
    # Rewrite call keeping in mind d instead of hidden_dim    
    def __call__(self, inputs):
        msg, node = inputs
        msg = tf.reshape(msg, [self.batch_size, 1, self.d*self.n])
        node = tf.reshape(node, [self.batch_size, self.d*self.n])
        print("msg: ", msg.shape)
        print("node: ", node.shape)
        node_next, state = self.gru(msg, initial_state = node)
        # node_next = tf.reshape(node_next, [self.batch_size, self.n_node, self.hidden_dim])
        return node_next, state
    
class msgFunc_NNforEN(Layer):
    def __init__(self, edge_dim, **kwargs):
        super(msgFunc_NNforEN, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.d = 12
        x = Input(shape=(self.edge_dim, ))
        x1 = Dense(self.d*self.d, activation = 'relu')(x)
        out = Reshape((self.d, self.d))(x1)
        self.model = tf.keras.Model(inputs=x, outputs=out, name='Edge Preprocessing')

    def __call__(self, adjMat):
        # Likely need to check its dimension
        pass

        
# Implementation of Matrix Multiplication using Edge Networks in page 5
class msgFunc_EN(Layer):
    def __init__(self, batch_size, n_node, edge_dim, d, **kwargs):
        super(msgFunc_EN, self).__init__(**kwargs)
        self.d = d
        self.n_node = n_node
        self.edge_dim = edge_dim
        self.batch_size = batch_size
        self.m_in = msgFunc_NNforEN(self.edge_dim)
        self.m_out = msgFunc_NNforEN(self.edge_dim)

    # Accepts the adjacency matrix of the whole batch and iterates over the molecules
    def process(self, adj_mat):
        edge_val_in = np.zeros((self.batch_size, self.n_node, self.n_node, self.d, self.d))
        edge_val_out = np.zeros((self.batch_size, self.n_node, self.n_node, self.d, self.d))
        
        # Loop to iterate over the molecules
        for i in range(len(adj_mat)):
            a_in_mat = np.zeros((self.n_node, self.n_node, self.d, self.d))
            a_out_mat = np.zeros((self.n_node, self.n_node, self.d, self.d))
            for v in range(self.n_node):
                for w in range(self.n_node):
                    a_in = self.m_in(adj_mat[i][v][w])
                    # a_in = tf.reshape(a_in, [self.d, self.d])
                    a_out = self.m_out(adj_mat[i][v][w])
                    # a_out = tf.reshape(a_out, [self.d, self.d])

                    a_in_mat[v][w] = a_in
                    a_out_mat[v][w] = a_out
            
            edge_val_in[i] = a_in_mat
            edge_val_out[i] = a_out_mat
        return edge_val_in, edge_val_out


    def __call__(self, h, A):
        # A: d x d, h: d x n
        msg = tf.matmul(A, h)
        return msg

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