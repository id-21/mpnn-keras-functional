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