import numpy as np 
import random
from scipy.sparse import issparse 
from numpy.linalg.linalg import norm
from numpy.random import randint

class SKMeans:
    def __init__(self, no_clusters, iters=300):
        self.no_clusters = no_clusters
        self.iters = iters

    def spherical_clustering(self, input_matrix):
        sample_count = input_matrix.shape[1]
        labels = np.zeros(sample_count, dtype='int8')
        input_matrix = input_matrix/(norm(input_matrix,axis=1).reshape(sample_count,1))     
        iters = 0
        while iters < self.iters:
            print iters
            self.centres = self.centres/(norm(self.centres,axis=1).reshape(self.no_clusters,1))
            clusdotprod = input_matrix.dot(self.centres.T)
            for i in xrange(sample_count):
                labels[i] = np.argmax(clusdotprod[i,:])            
            clus_old = self.centres.copy()
            for i in unique(labels):
                self.centres[i,:] = mean(bnorm[labels==i], axis=0)
            iters += 1


    def run_kmeans(self, input_matrix, delta=.001):
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
            if (1 - delta) * prev_distance <= avg_distance <= prev_distance: break
            prev_distance = avg_distance
            print prev_distance
            for label in xrange(self.no_clusters):
                indexes = np.where(self.labels == label)[0]
                if len(indexes) > 0:
                    self.centres[label] = input_matrix[indexes].mean(axis=0)

    def sample_centres(self, input_matrix):
        sampleidx = random.sample(xrange(input_matrix.shape[0]), int(self.no_clusters))
        return input_matrix[sampleidx]

    def fit(self, input_matrix, sample_centres=True, param_centres=None):
        if sample_centres:
            self.centres = input_matrix[randint(0,input_matrix.shape[0],self.no_clusters)]
        else:
            if not param_centres:
                raise ValueError("Must provide centre matrix if sample_centres is set to False.")
            self.centres = param_centres
        self.run_kmeans(input_matrix)

    def get_labels(self):
        return self.labels

    def get_distances(self):
        return self.distances

    def get_centres(self):
        return self.centres


