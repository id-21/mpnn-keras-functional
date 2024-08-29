import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input


class set2set(Layer):
    def __init__(self, T, n_hidden, **kwargs):
        super(set2set, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.T = T
        self.state_size = [n_hidden, n_hidden]  # LSTM has two states: hidden state and cell state
        self.recurrent_activation = K.sigmoid
        self.activation = K.tanh

    def build(self, input_shape):
        feat_dim = input_shape[-1]

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


N_HIDDEN = 10
T = 4
FEAT_DIM = 7
BATCH_SIZE = 5
N_FEAT = 6

lay = set2set(T, N_HIDDEN)
inputs = Input(shape = (N_FEAT, FEAT_DIM), batch_size=BATCH_SIZE)
out = lay(inputs)

model = tf.keras.Model(inputs = inputs, outputs = out)
model.summary()