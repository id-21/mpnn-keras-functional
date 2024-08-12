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
   

# Function that returns a GRU Layer for the message update step.
# Called in the msgFunc_EN.call().
class upFunc_GRU(Layer):
    def __init__(self, batch_size, n_node, d, **kwargs):
        super(upFunc_GRU, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n = n_node
        self.d = d
        self.n_node = n_node

        self.gru = GRU(self.d * self.n, return_sequences=True, return_state=True)
    
    def call(self, inputs):
        msg, node = inputs
        msg = tf.reshape(msg, [self.batch_size, 1, self.d*self.n])
        node = tf.reshape(node, [self.batch_size, self.d*self.n])
        print("msg: ", msg.shape)
        print("node: ", node.shape)
        node_next, state = self.gru(msg, initial_state = node)
        # node_next = tf.reshape(node_next, [self.batch_size, self.n_node, self.hidden_dim])
        return node_next, state

# Function that returns a NN model that can preprocess the edge features. 
# Called in msgFunc_EN.__init__() and msgFunc_EN.process() functions.   
class msgFunc_NNforEN(Layer):
    def __init__(self, edge_dim, d, **kwargs):
        super(msgFunc_NNforEN, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.d = d
        x = Input(shape=(self.edge_dim,))
        x1 = Dense(self.d*self.d, activation = 'relu')(x)
        out = Reshape((self.d, self.d))(x1)
        self.model = tf.keras.Model(inputs=x, outputs=out, name='EdgePreprocessing')

    def call(self, adjMat):
        # Likely need to check its dimension
        return self.model(adjMat)


# Custom layer to process each edge
class EdgeProcessingLayer(Layer):
    def __init__(self, edge_model, batch_size, n_node, edge_dim):
        super(EdgeProcessingLayer, self).__init__()
        self.edge_model = edge_model
        self.batch_size = batch_size
        self.n_node = n_node
        self.edge_dim = edge_dim

    def call(self, inputs):
        # Flatten the input tensor to apply the model
        flattened_input = tf.reshape(inputs, (-1, self.edge_dim))
        # Process each edge
        processed = self.edge_model(flattened_input)
        output_dim = tf.shape(processed)[-1]
        reshaped_output = tf.reshape(processed, (self.batch_size, self.n_node, self.n_node, output_dim, output_dim))
        return reshaped_output


# Implementation of Matrix Multiplication using Edge Networks in page 5
class msgFunc_EN(Layer):
    def __init__(self, batch_size, n_node, edge_dim, d, **kwargs):
        super(msgFunc_EN, self).__init__(**kwargs)
        self.d = d
        self.n_node = n_node
        self.edge_dim = edge_dim
        self.batch_size = batch_size
        self.m_in = msgFunc_NNforEN(self.edge_dim, self.d)
        self.A_in = np.zeros((self.batch_size, self.n_node, self.n_node, self.d, self.d))

    # Accepts the adjacency matrix of the whole batch and iterates over the molecules
    def process(self, adj_mat):
        processed_edges = EdgeProcessingLayer(self.m_in, self.batch_size, self.n_node, self.edge_dim)(adj_mat)
        return processed_edges
    
    '''
    The call function of this message computing function is defined to allow batch processing of 
    molecules. Details regarding the implementation are explained in model_arch.md

    Call arguments:
    h: node matrix for which messages are to be computed
    
    The function has been hard coded to compute the message for only the first element of the batch
    '''
    def call(self, h):
        # A: (batch_size, n_node, n_node, d, d);
        # h: (batch_size, d, n)
        A = self.A_in
        # Converting the data type of A to float32
        A = tf.reshape(A, [self.batch_size * self.n_node, self.n_node * self.d, self.d], name='FeaturizedEdgeMatrix')
        A = tf.cast(A, dtype=tf.float32)

        # Converting the data type of A to float32
        h_ = tf.reshape(h, [self.batch_size * self.n_node, self.d, 1], name='NodeRepresentation')
        h_ = tf.cast(h_, dtype=tf.float32)

        msg = tf.matmul(A, h_)
        msg = tf.reshape(msg, [self.batch_size, self.n_node, self.n_node, self.d])
        # To reduce the dimension of this matrix, we do a reshuffle and then a reduce_mean operation
        msg = tf.transpose(msg, perm = [0, 2, 3, 1])
        msg = tf.reduce_mean(msg, 3)
        return msg

def test(n_step, batch_size, n_node, hidden_dim):
    pass

def create_mpnn_model(n_step, batch_size, n_node, d, edge_dim):
    edge_wt_input = Input(shape=(n_node, n_node, edge_dim), batch_size=batch_size, name='edge_wt')
    node_hidden_input = Input(shape=(n_node, d), batch_size=batch_size, name='node_hidden')
    # mask_input = Input(shape=(n_node, hidden_dim), batch_size=batch_size, name='mask')
    edge_net = msgFunc_EN(batch_size, n_node, edge_dim, d)
    upd_GRU = upFunc_GRU(batch_size, n_node, d)

    nhs = []
    adjM = edge_net.process(edge_wt_input)
    for _ in range(n_step):
        msg = edge_net(node_hidden_input)
        node_hidden, state = upd_GRU.call([msg, node_hidden_input])
        nhs.append(node_hidden)
    print(out.shape)
    out = tf.concat(nhs, axis = 2)
    model = tf.Model(inputs=[edge_wt_input, node_hidden_input], outputs=out)
    return model