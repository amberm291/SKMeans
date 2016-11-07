import numpy as np 
import random
from scipy.sparse import issparse 
from numpy.linalg.linalg import norm
from numpy.random import randint

class SKMeans:
    def __init__(self, no_clusters, iters=300):
        '''
            Constructor for the class.

            PARAMETERS:
                no_clusters (int): The number of clusters to be generated.
                iters (int): Number of iterations for the algorithm.
        '''
        self.no_clusters = no_clusters
        self.iters = iters

    def run_kmeans(self, input_matrix, delta=.001):
        '''
            The actual function performing kmeans.

            PARAMETERS:
                input_matrix (scipy.sparse or numpy.ndarray): Matrix containing input sample. It can either be a scipy sparse matrix or a numpy 2darray.
                delta (float): Stopping criteria for k-means
            
            RETURNS:
                None
        '''
        input_samples, input_dimensions = input_matrix.shape
        no_centres, centre_dimensions = self.centres.shape
        if input_dimensions != centre_dimensions:
            raise ValueError("Number of dimensions in input samples and centres should be same")
        prev_distance = 0
        input_seq = np.arange(input_samples)
        for i in xrange(self.iters):
            self.distances = input_matrix.dot(self.centres.T)
            if issparse(self.distances):
                self.distances = self.distances.toarray()
            self.labels = self.distances.argmax(axis=1)
            self.distances = np.ones(input_samples) - self.distances[input_seq,self.labels]
            avg_distance = self.distances.mean()
            try:
                if (1 - delta) * prev_distance <= avg_distance <= prev_distance: break
            except Exception, e:
                continue
            prev_distance = avg_distance
            for label in xrange(self.no_clusters):
                indexes = np.where(self.labels == label)[0]
                if len(indexes) > 0:
                    self.centres[label] = input_matrix[indexes].mean(axis=0)

    def sample_centres(self, input_matrix, no_samples):
        '''
            Function to sample centres from the input matrix.

            PARAMETERS:
                input_matrix (scipy.sparse or numpy.ndarray): Matrix containing input samples. It can either be a scipy sparse matrix or a numpy 2darray.
                no_samples (int): The number of points to be sampled from input_matrix.

            RETURNS:
                A sparse matrix or numpy 2darray (depending upon input) with number of rows equal to no_samples and number of columns equal to column of input_matrix.
        '''
        return input_matrix[randint(0,input_matrix.shape[0],no_samples)]

    def sample_kmeans(self, input_matrix):
        '''
            Two pass k-means to sample centres in the first pass by running k-means on a small input. The second pass uses these centres to compute the centres on the entire data.

            PARAMETERS:
                input_matrix (scipy.sparse or numpy.ndarray): Matrix containing input samples. It can either be a scipy sparse matrix or a numpy 2darray.

            RETURNS:
                None
        '''
        input_points, no_dimensions = input_matrix.shape
        no_samples = max(2*np.sqrt(input_points), 10*self.no_clusters)
        sampled_input = self.sample_centres(input_matrix, no_samples)
        self.centres = self.sample_centres(input_matrix, self.no_clusters)
        self.run_kmeans(sampled_input)
        self.run_kmeans(input_matrix)

    def fit(self, input_matrix, sample=True, param_centres=None, two_pass=False):
        '''
            Function to input the data and call run_kmeans on it.

            PARAMETERS:
                input_matrix (scipy.sparse or numpy.ndarray): Matrix containing input samples. It can either be a scipy sparse matrix or a numpy 2darray.
                sample (boolean): By default set to True, this flag is used to sample centres from the input data. If set to False, a numpy array containing centre points should be passed to param_centres.
                param_centres (scipy.sparse or numpy.ndarray): Is set to None by default. Should be passed a sparse matrix or numpy 2darray containing centre points, if sample is set to False.
                two_pass (boolean): By default set to Flase, set this flag to True to execute a two pass k-means. If set to True, this flag takes precedence over the sample flag and ignores its value.

            RETURNS: None

        '''
        input_matrix = input_matrix/norm(input_matrix,axis=1)

        if two_pass:
            self.sample_kmeans(input_matrix)
            return

        if sample:
            self.centres = self.sample_centres(input_matrix, self.no_clusters)
        else:
            if not param_centres:
                raise ValueError("Must provide centre matrix if sample_centres is set to False.")
            self.centres = param_centres/norm(param_centres,axis=1)
        self.run_kmeans(input_matrix)

    def get_labels(self):
        '''
            Function to get cluster labels for each point in the input matrix.

            PARAMETERS:
                None 

            RETURNS:
                labels (list of ints): List containing labels for each point in the same order as they were passed in input matrix.
        '''
        return self.labels

    def get_distances(self):
        '''
            Function to get distances for each point from their cluster centre in the input matrix.

            PARAMETERS:
                None 

            RETURNS:
                distances (np.array of type np.float64): Numpy array containing distances for each point in the same order as they were passed in input matrix.
        '''
        return self.distances

    def get_centres(self):
        '''
            Function to get centre arrays for each cluster.

            PARAMETERS:
                None 

            RETURNS:
                centres (numpy.ndarray): Numpy 2darray where each represents the centre of a cluster.
        '''
        return self.centres


