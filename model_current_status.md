PROGRESS (WORK DONE):
1. Functions for each steps are ready: 
	a. class msgFunc_NNforEN(Layer): Builds the Neural Network for Edge Featurisation
	b. class msgFunc_EN(Layer): 
		i. In its process() function, the adjacency matrix is taken and the edge vectors are featurised. This is the preprocessing step to make the message passing steps faster.
		ii. In its call() function, the vectorised Edge Network is multiplied with the node representation to compute the messages. These messages are what will be used for the GRU Update step. 

---- TO BE COMPLETED ----

CURRENT ERROR: 
Traceback (most recent call last):
  File "C:\Users\ok\mpnn_functional\mpnn-keras-functional\_model_func_api.py", line 51, in <module>
    model = create_mpnn_model(n_step, batch_size, n_node, d, edge_dim)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ok\mpnn_functional\mpnn-keras-functional\_model_func.py", line 164, in create_mpnn_model
    adjM = edge_net.process(edge_wt_input)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ok\mpnn_functional\mpnn-keras-functional\_model_func.py", line 109, in process
    A = self.m_in(tf.reshape(adj_mat[i][v][w], (1, self.edge_dim)))
                             ~~~~~~~^^^
  File "C:\Users\ok\anaconda3\Lib\site-packages\keras\src\backend\common\keras_tensor.py", line 290, in __getitem__
    return ops.GetItem().symbolic_call(self, key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ok\anaconda3\Lib\site-packages\keras\src\ops\operation.py", line 60, in symbolic_call
    outputs = self.compute_output_spec(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ok\anaconda3\Lib\site-packages\keras\src\ops\numpy.py", line 2584, in compute_output_spec
    raise ValueError(
ValueError: Unsupported key type for array slice. Recieved: 0_
__________________________________________________________________________
Reason for error:
The code works for predefined input Tensors but does not produce output for the Eager tensor. 
The key type also refers to the batch_size dimension which also points to a problem with Eager Tensor code execution. 


NEW ERROR: 

File "C:\Users\ok\mpnn_functional\mpnn-keras-functional\_model_func.py", line 113, in process
    adj_value = tf.gather_nd(adj_mat, [index])  # Shape: (edge_dim,)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


ChatGPT says this will work:
 def process(self, adj_mat):
        # Initialize an empty list to store processed features
        edge_val_list = []

        # Iterate over the batch
        for i in range(self.batch_size):
            A_batch_list = []  # List to hold processed features for a single batch
            
            # Iterate over nodes
            for v in range(self.n_node):
                for w in range(self.n_node):
                    # Use Keras operations to gather the adjacency value
                    adj_value = tf.expand_dims(adj_mat[i, v, w], axis=0)  # Shape: (1, edge_dim)

                    # Process each element using the neural network layer
                    A = self.m_in(adj_value)  # Shape: (1, d * d)

                    # Reshape A to the desired shape
                    A = tf.reshape(A, (self.d, self.d))

                    # Append the processed feature to the list
                    A_batch_list.append(A)
            
            # Stack the processed features and reshape
            A_batch_tensor = tf.stack(A_batch_list, axis=0)
            A_batch_tensor = tf.reshape(A_batch_tensor, (self.n_node, self.n_node, self.d, self.d))

            # Append to the edge_val_list
            edge_val_list.append(A_batch_tensor)

        # Stack the final tensor
        edge_val_in_tensor = tf.stack(edge_val_list, axis=0)
        self.A_in = edge_val_in_tensor

        return edge_val_in_tensor

