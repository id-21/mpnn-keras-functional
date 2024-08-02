The MPNN model proceeds as follows:
1. Each graph is preprocessed to make the message passing steps faster. The edges of the graph are represented using vectors. These vectors are embedded using a neural network in a d x d dimensional matrix. (d is the dimension of the internal node representation of each node)
2. Next, the messages are computed by performing a matrix multiplication between the vector embeddings and the node vectors (d x 1 dimensional matrices). 
3. The node representations are then updated using a Gated Recurrent Unit (GRU) with the messages as the input and the initial state as the initial node representation.
4. Message passing is performed for t steps. 
5. The final node representations are then fed into another neural netowrk to make predictions about molecular properties.  


Shapes of the various variables:
node_inp: Input node feature vectors 
	Shape -> (batch_size, n_node, node_dim)
(h_v)^t: node feature vectors after the t-th message passing step
(h_v)^0: node feature vectors after the 0-th message passsing step
	Shape -> (batch_size, n_node, d)
		where d > node_dim
adj_mat: adjacency matrix of the molecular graph, each element is a vector with the bond distance and the bond type encoded as a OHE
	Shape -> (batch_size, n_node, n_node, edge_dim)
		where edge_dim = 1 + no_of_bond_types
A_in, A_out: processed adjacency matrices where each element is a 2D matrix
	Shape -> (batch_size, n_node, n_node, d, d)
msg: Message computed
	Shape -> (batch_size, n_node, n_node, d)
	During message update, values across the 3rd dimension would need to be collated using a function to prepare the input for the GRU Layer ( Shape -> (batch_size, n_node, d) )


To process and compute the messages of single molecules in a single batch, there is a challenge. While for a single molecule, the dimensions match perfectly for a A(wv) * (h_v)^t operation, given the rules of matrix multiplication, this cannot be done for our 5D and 3D matrices of A and h_v respectively. 
A workaround is dimension mixing: we reshape the matrices to multiplication compatible sizes:

(h_v)^t: Shape -> (batch_size, n_node, d) is converted to (batch_size * n_node, d, 1)
A_in: Shape -> (batch_size, n_node, n_node, d, d) is converted to (batch_size * n_node, n_node * d, d)

Thus, after the operation, we get a matrix of shape (batch_size * n_node, n_node * d, 1) which can be reshaped to (batch_size, n_node, n_node, d)

This can be expected to represent the matrix containing the messages recieved by the node index of the first dimension by the node index of the second dimension. We can then collate the messages across the third dimension to get the message that needs to be updated to the node features using the GRU. 

