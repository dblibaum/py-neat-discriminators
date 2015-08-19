"""Evolves discriminating functions for data processed with make_discriminator_training_set.

   Use: Initialize with data set, then call evolve() with number of generations and max number of features to evolve.
        Saves the discriminator networks to /networks.

"""

import MultiNEAT as NEAT
import numpy as np
from numpy import random, argsort
import time
import scipy.spatial.ckdtree as kd


class EvolveDiscriminators:

    def __init__(self, pairs, max_features):
        """
        :param pairs: :type iterable: Data the discriminators are to be trained on. Should be processed by
                                      make_discriminator_training_set.
        """

        # Input data
        self.pairs = pairs
        self.dsize = len(pairs)

        self.esize = len(pairs[0])

        # Maintain novelty archive and two separate lists for the behavior vectors and genomes of features
        self.novelty_archive = np.zeros((500, self.dsize))  # Space for 500 points. Make this a bit bigger then n_gen
        self.feature_list_vectors = np.zeros((max_features, self.dsize))  # Space for max_features
        self.feature_list_genomes = []
        self.max_features = max_features
        self.archive_size = 0

        # NEAT initializations
        self.params = NEAT.Parameters()
        self.params.PopulationSize = 100

        self.genome = NEAT.Genome(0, self.esize + 1, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                                  NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, self.params)

        self.pop = NEAT.Population(self.genome, self.params, True, 1.0)

    def get_novelty(self, vector_space, genome_vector):
        """Return the novelty for a single behavior vector.

        :param vector_space: :type list: A list of lists representing the whole behavior space.
        :param genome_vector: :type list: The behavior vector of the genome to be evaluated.
        :return: Novelty score
        """

        novelty = 0

        # Get space of generation + novelty archive
        vector_space = np.concatenate((vector_space, self.novelty_archive))

        # Get distances to k-nearest points in behavior space
        tree = kd.cKDTree(vector_space)
        indexes = tree.query(genome_vector, 20)
        distances = indexes[0]

        for i in distances:
            novelty += i/self.dsize
        novelty /= 20

        # Add to novelty archive with 1% probability
        rand_n = random.random()
        if rand_n >= .99:
            k = 0
            for element in genome_vector:
                self.novelty_archive[self.archive_size][k] = element
                k += 1
            self.archive_size += 1

        return novelty

    def get_output_vector(self, genome):
        """Returns the behavior vector for a single genome.

        :param genome: :type NEAT.genome: The genome to be evaluated.
        :return o_vector: :type list: A vector representing the behavior of the genome.
        """

        o_vector = np.zeros(self.dsize)

        i = 0
        for pair in self.pairs:

            # Bias
            pair.append(1)

            # This creates a neural network (phenotype) from the genome
            net = NEAT.NeuralNetwork()
            genome.BuildPhenotype(net)

            # Input just one pattern to the net, activate it once and get the output
            net.Input(pair)
            net.Activate()
            o_vector[i] = net.Output()[0]
            i += 1

        return o_vector

    def evolve(self, n_gen):
        """Perform evolution of the discriminators on the data provided in the initializer. Saves the discriminator
           networks to /networks.

        :param n_gen: :type int: Number of generations to run the evolution.
        """

        try:
            for generation in range(n_gen):  # run for n_gen generations

                if len(self.feature_list_genomes) < self.max_features:

                    start = time.clock()
                    outputs = np.zeros((self.params.PopulationSize, self.dsize))

                    # retrieve a list of all genomes in the population
                    genome_list = NEAT.GetGenomeList(self.pop)

                    # Get output space vectors for all genomes
                    j = 0
                    for genome in genome_list:
                        print "Computing behavior vector for genome " + str(j)
                        output_array = self.get_output_vector(genome)
                        k = 0
                        for element in output_array:
                            outputs[j][k] = element
                            k += 1
                        j += 1

                    # Evaluate novelty for all genomes and add to feature list
                    i = 0
                    for genome in genome_list:
                        print "Computing novelty for genome " + str(i)
                        genome_vector = outputs[i]
                        fitness = self.get_novelty(outputs, genome_vector)

                        # Add to feature list
                        if len(self.feature_list_genomes) == 0:
                            m = 0
                            for element in genome_vector:
                                self.feature_list_vectors[len(self.feature_list_genomes)][m] = element
                                m += 1
                            self.feature_list_genomes.append(genome)
                        else:
                            tree = kd.cKDTree(self.feature_list_vectors)
                            indexes = tree.query(genome_vector, 20)
                            dist = indexes[0][0]
                            print "Distance from closest in feature list: " + str(dist)
                            if dist > 200:
                                m = 0
                                for element in genome_vector:
                                    self.feature_list_vectors[len(self.feature_list_genomes)][m] = element
                                    m += 1
                                self.feature_list_genomes.append(genome)

                        genome.SetFitness(fitness)
                        i += 1

                    n = 0
                    for genome in self.feature_list_genomes:
                        net = NEAT.NeuralNetwork()
                        genome.BuildPhenotype(net)
                        net.Save("/home/dan/solus/solus_design/constructor/networks/net" + str(n) + ".nnet")
                        n += 1

                    # datafile = open("discriminator_genomes.pkl", "wb")
                    # pickle.dump(self.feature_list_genomes, datafile)
                    # datafile.close()
                    # datafile = open("discriminator_genomes_backup.pkl", "wb")
                    # pickle.dump(self.feature_list_genomes, datafile)
                    # datafile.close()

                    end = time.clock()
                    elapsed = end - start

                    print str(generation) + " generations evaluated, " + str(elapsed) + " for last gen."
                    print str(len(self.feature_list_genomes)) + " discriminators evolved."
                    print "----------"

                    # advance to the next generation
                    self.pop.Epoch()
                else:
                    break
        except KeyboardInterrupt:
            print "Training stopped."
