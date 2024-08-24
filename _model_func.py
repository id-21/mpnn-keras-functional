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
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Reshape, Lambda, Layer, multiply, TimeDistributed
from tensorflow.keras.models import Sequential

def f(x):
    print("Here s", x[0][0])
    return (x[0][0], x[0][1], x[0][2], x[0][3], 1)

matmul_layer = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]), output_shape = f)

class upFunc_GRU(Layer):
    def __init__(self, batch_size, n_node, d, **kwargs):
        super(upFunc_GRU, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n = n_node
        self.d = d
        self.n_node = n_node

        self.gru = GRU(self.d * self.n, return_sequences=False, return_state=True)
      
    def call(self, inputs):
        msg, node = inputs
        msg = Reshape((1, self.d*self.n, ))(msg)
        node = Reshape((self.d*self.n, ))(node)
        node_next= self.gru(msg, initial_state = node)
        return node_next
    
class msgFunc_NNforEN(Layer):
    def __init__(self, edge_dim, d, **kwargs):
        super(msgFunc_NNforEN, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.d = d
        x = Input(shape=(self.edge_dim,))
        x1 = Dense(self.d*self.d, activation = 'relu')(x)
        out = Reshape((self.d, self.d))(x1)
        self.model = tf.keras.Model(inputs=x, outputs=out, name='Edge Processing NN')

    def compute_output_shape(self, input_shape):
      return (input_shape[0], self.d*self.d)

    def call(self, adjMat):
        return self.model(adjMat)
      
# Implementation of Matrix Multiplication using Edge Networks in page 5
class msgFunc_EN(Layer):
    def __init__(self, batch_size, n_node, edge_dim, d, **kwargs):
        super(msgFunc_EN, self).__init__(**kwargs)
        self.d = d
        self.n_node = n_node
        self.edge_dim = edge_dim
        self.batch_size = batch_size
        self.m_in = msgFunc_NNforEN(self.edge_dim, self.d)

    def call(self, adj_mat):
        adj_3D = Reshape((-1, self.edge_dim))(adj_mat)
        vec_adj_3D = TimeDistributed(self.m_in)(adj_3D)
        print(vec_adj_3D.shape)
        final_adj = Reshape((self.n_node, self.n_node, self.d, self.d))(vec_adj_3D)
        self.A_in = final_adj
        return final_adj

def create_mpnn_model(n_step, BATCH_SIZE, N_NODE, D, EDGE_DIM):
    edge_wt_input = Input(shape=(N_NODE, N_NODE, EDGE_DIM), batch_size=BATCH_SIZE, name='edge_wt')
    h_0 = Input(shape=(N_NODE, D, 1), batch_size=BATCH_SIZE, name='node_feat_0')
    # mask_input = Input(shape=(n_node, hidden_dim), batch_size=batch_size, name='mask')
    edge_net = msgFunc_EN(BATCH_SIZE, N_NODE, EDGE_DIM, D, name='EdgeNeuralNetwork')
    upd_GRU = upFunc_GRU(BATCH_SIZE, N_NODE, D, name='msgNodeUpdateGRU')
    A = edge_net.call(edge_wt_input)
    print("Done!")


    h_ = Reshape((N_NODE, D, 1), name = 'reshape_h')(h_0) # 4-D tensor
    A_ = Reshape((N_NODE, N_NODE, D, D), name = 'reshape_A')(A) # 5-D tensor
    nhs = []
    # adjM = edge_net.process(edge_wt_input)
    for i in range(n_step):
        # msg = edge_net(node_hidden_input)
        output_shape = (N_NODE, N_NODE, D, 1)
        print(A_.shape)
        msg = matmul_layer([A_, h_])
        # Calculate the average along the second dimension (axis=1)
        msg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(msg)
        msg = Reshape((1, N_NODE*D), name = f'messageReshape{i}')(msg)
        node = Reshape((N_NODE*D, ), name = f'nodeReshape{i}')(h_)
        _, h_ = upd_GRU([msg, node])
        print(h_.shape)

    # out = tf.concat(nhs, axis = 2)
    model = tf.keras.Model(inputs=[edge_wt_input, h_0], outputs=h_)
    return model