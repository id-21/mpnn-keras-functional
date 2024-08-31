# PENDING TASKS
# 1. Figure out how to apply masks using Keras Layers Embedding OR fix Keras.Multiply used to apply mask
# https://www.tensorflow.org/guide/keras/understanding_masking_and_padding

# Must set these before importing tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
from tensorflow.keras import backend as K
tf.autograph.set_verbosity(0)

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Reshape, Lambda, Layer, multiply, TimeDistributed
from tensorflow.keras.models import Sequential
from einops import einsum

def f(x):
    return (x[0][0], x[0][1], x[0][2], x[0][3], 1)

def g(x):
    return (x[0][0], len(x), x[0][1])

# matmul_layer = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]), output_shape = f, name = 'MatMulLayer')
matmul_layer = tf.keras.layers.Lambda(lambda x: einsum(x[0], x[1], "a b c d e, a f d g -> a b c d g"), output_shape = f, name = 'MatMulLayer')

concat_layer = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1), output_shape = g, name = 'ConcatLayer')


class upFunc_GRU(Layer):
    def __init__(self, batch_size, n_node, d, **kwargs):
        super(upFunc_GRU, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n = n_node
        self.d = d
        self.n_node = n_node

        self.gru = GRU(self.d * self.n, return_sequences=False, return_state=True)
      
    def build(self, input_shape):
        # self.gru.build()
        self.gru.build(input_shape)
        super(upFunc_GRU, self).build(input_shape)

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
        self.model = tf.keras.Model(inputs=x, outputs=out, name='EdgeProcessingNN')

    def build(self, input_shape):
        super(msgFunc_NNforEN, self).build(input_shape)

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
        vec_adj_3D = TimeDistributed(self.m_in, name = 'EdgeNeuralNetworkTD')(adj_3D)
        final_adj = Reshape((self.n_node, self.n_node, self.d, self.d), name = 'adjMatReshape')(vec_adj_3D)
        self.A_in = final_adj
        return final_adj

'''
Takes as input, T and n_hidden. n_hidden is the number of units in the
LSTM. Returns a single vector that is of size 2*n_hidden
Reads number of input features from the input dynamically
'''
class set2set(Layer):
    def __init__(self, T, n_hidden, **kwargs):
        super(set2set, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.T = T
        self.state_size = [n_hidden, n_hidden]  # LSTM has two states: hidden state and cell state
        self.recurrent_activation = K.sigmoid
        self.activation = K.tanh

    def build(self, input_shape):
        # print(input_shape)
        feat_dim = input_shape[-1]
        # print(feat_dim)
        # Initialise weight for linear projection
        self.kernel = self.add_weight(shape=(feat_dim, self.n_hidden),
                                    initializer='glorot_uniform',
                                    name='kernel')

        # Initialize weights for input gate
        self.W_i = self.add_weight(shape=(feat_dim + self.n_hidden, self.n_hidden),
                                   initializer='glorot_uniform',
                                   name='W_i')
        self.b_i = self.add_weight(shape=(self.n_hidden,),
                                   initializer='zeros',
                                   name='b_i')
        # Initialize weights for forget gate
        self.W_f = self.add_weight(shape=(feat_dim + self.n_hidden, self.n_hidden),
                                   initializer='glorot_uniform',
                                   name='W_f')
        self.b_f = self.add_weight(shape=(self.n_hidden,),
                                   initializer='zeros',
                                   name='b_f')
        # Initialize weights for output gate
        self.W_o = self.add_weight(shape=(feat_dim + self.n_hidden, self.n_hidden),
                                   initializer='glorot_uniform',
                                   name='W_o')
        self.b_o = self.add_weight(shape=(self.n_hidden,),
                                   initializer='zeros',
                                   name='b_o')
        # Initialize weights for cell state
        self.W_c = self.add_weight(shape=(feat_dim + self.n_hidden, self.n_hidden),
                                   initializer='glorot_uniform',
                                   name='W_c')
        self.b_c = self.add_weight(shape=(self.n_hidden,),
                                   initializer='zeros',
                                   name='b_c')
        super(set2set, self).build(input_shape)
    
    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        n_features = K.shape(inputs)[1]

        # Initialize the hidden state (h_t) and cell state (c_t) for the LSTM
        h_t = K.zeros((batch_size, self.n_hidden))
        c_t = K.zeros((batch_size, self.n_hidden))

        q_star = K.zeros((batch_size, 2 * self.n_hidden))

        for _ in range(self.T):
            # Repeat hidden state across all features
            h_t_expanded = K.repeat(h_t, n_features)
            
            # Compute attention weights
            m_t = K.dot(inputs, self.kernel)  # Linear transformation
            e_t = K.sum(m_t * h_t_expanded, axis=-1)  # Attention scores
            a_t = K.softmax(e_t)  # Attention weights
            
            # Compute context vector r_t
            a_t_expanded = K.expand_dims(a_t, axis=-1)
            r_t = K.sum(a_t_expanded * inputs, axis=1)
            
            # Update q_star
            q_star = K.concatenate([h_t, r_t], axis=-1)
            
            # Update hidden state h_t and cell state c_t using LSTM equations
            h_t, c_t = self._lstm_step(q_star, h_t, c_t)

        return q_star

    def _lstm_step(self, x, h_tm1, c_tm1):
        # Input gate
        i = self.recurrent_activation(K.dot(inputs, self.W_i) + K.dot(h_prev, self.U_i) + self.b_i)
        
        # Forget gate
        f = self.recurrent_activation(K.dot(inputs, self.W_f) + K.dot(h_prev, self.U_f) + self.b_f)
        
        # Output gate
        o = self.recurrent_activation(K.dot(inputs, self.W_o) + K.dot(h_prev, self.U_o) + self.b_o)
        
        # Cell state
        c_tilde = self.activation(K.dot(inputs, self.W_c) + K.dot(h_prev, self.U_c) + self.b_c)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # New hidden state
        h = o * self.activation(c)
        
        return h, c

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.n_hidden)

class MPNN(Layer):
    def __init__(self, batch_size, n_node, d, n_step, **kwargs):
        super(MPNN, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.n_node = n_node
        self.d = d
        self.n_step = n_step
        self.upd_GRU = upFunc_GRU(self.batch_size, self.n_node, self.d, name='msgNodeUpdateGRU')

    def build(self, input_shape):
        self.upd_GRU.build(input_shape)
        super(MPNN, self).build(input_shape)


    def call(self, A_, h_0):
        h_ = Reshape((self.n_node, self.d, 1), name = 'Input_reshape')(h_0) # 4-D tensor
        nhs = []

        for i in range(self.n_step):
            output_shape = (self.n_node, self.n_node, self.d, 1)
            # print([h_, A_])
            msg = matmul_layer([A_, h_])
            # Calculate the average along the second dimension (axis=1)
            msg = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name = f'MsgAvgLayer{i}')(msg)
            msg = Reshape((1, self.n_node*self.d), name = f'messageReshapeforGRU{i}')(msg)
            node = Reshape((self.n_node*self.d, ), name = f'nodeReshapeforGRU{i}')(h_)
            _, h_ = self.upd_GRU([msg, node])
            nhs.append(h_)

        stacked_nhs = concat_layer(nhs)
        return stacked_nhs

    def compute_output_shape(self, input_shape):
        out_shape = (self.batch_size, self.n_step+1, self.n_node*self.d)
        return out_shape

def create_mpnn_model(N_STEP, BATCH_SIZE, N_NODE, D, EDGE_DIM, T, N_HIDDEN, PROP_D):
    # Layer Definitions
    edge_net = msgFunc_EN(BATCH_SIZE, N_NODE, EDGE_DIM, D, name='EdgeNeuralNetwork')
    mpnn = MPNN(BATCH_SIZE, N_NODE, D, N_STEP, name='MPNN')
    # Input Definitions
    edge_wt_input = Input(shape=(N_NODE, N_NODE, EDGE_DIM), batch_size=BATCH_SIZE, name='edge_wt')
    h_0 = Input(shape=(N_NODE, D, 1), batch_size=BATCH_SIZE, name='node_feat_0')
    # Model Definition: Functional API
    A_ = edge_net.call(edge_wt_input)
    stacked_nhs = mpnn(A_, h_0)
    s2s = set2set(T, N_HIDDEN)
    out = s2s(stacked_nhs)
    pred = Dense(PROP_D)(out)

    model = tf.keras.Model(inputs=[edge_wt_input, h_0], outputs=pred)
    return model