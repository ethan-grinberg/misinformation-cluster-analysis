import stellargraph as sg
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras

class UGraphEmb:
    def __init__(self, num_pairs=100, 
                layer_sizes=[64,32], 
                activation_func='relu', 
                pool_all_layers=True,
                batch_size=10,
                epochs=500,
                verbose=0):
        self.a_func = activation_func
        self.num_pairs = num_pairs
        self.layer_sizes = layer_sizes
        self.pool_all_layers = pool_all_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, nx_graphs):
        graphs = self.__convert_sg(nx_graphs)

        # create model
        generator = sg.mapper.PaddedGraphGenerator(graphs)
        gc_model = sg.layer.GCNSupervisedGraphClassification(
            self.layer_sizes, [self.a_func, self.a_func], generator, pool_all_layers=self.pool_all_layers
        )

        inp1, out1 = gc_model.in_out_tensors()
        inp2, out2 = gc_model.in_out_tensors()
        vec_distance = tf.norm(out1 - out2, axis=1)
        pair_model = keras.Model(inp1 + inp2, vec_distance)
        embedding_model = keras.Model(inp1, out1)

        # training
        graph_idx = np.random.RandomState(0).randint(len(graphs), size=(self.num_pairs, 2))
        targets = [self.__graph_distance(nx_graphs[left], nx_graphs[right]) for left, right in graph_idx]
        train_gen = generator.flow(graph_idx, batch_size=self.batch_size, targets=targets)

        # training procedure
        pair_model.compile(keras.optimizers.Adam(1e-2), loss="mse")
        history = pair_model.fit(train_gen, epochs=self.epochs, verbose=self.verbose)

        # compute embeddings
        self.embeddings = embedding_model.predict(generator.flow(graphs))

    def get_embedding(self):
        return self.embeddings

    def __convert_sg(self, nx_graphs):
        graphs = []
        for g in nx_graphs:
            graphs.append(sg.StellarDiGraph(graph=g, node_features='out-degree'))
        
        return graphs
    
    def __graph_distance(self, g1, g2):
        return nx.graph_edit_distance(g1, g2)
