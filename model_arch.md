The MPNN model proceeds as follows:
1. Each graph is preprocessed to make the message passing steps faster. The edges of the graph are represented using vectors. These vectors are embedded using a neural network in a d x d dimensional matrix. (d is the dimension of the internal node representation of each node)
2. Next, the messages are computed by performing a matrix multiplication between the vector embeddings and the node vectors (d x 1 dimensional matrices). 
3. The node representations are then updated using a Gated Recurrent Unit (GRU) with the messages as the input and the initial state as the initial node representation.
4. Message passing is performed for t steps. 
5. The final node representations are then fed into another neural netowrk to make predictions about molecular properties.  