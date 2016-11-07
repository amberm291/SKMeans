## SKMeans

Implementation of k-means with cosine distance as the distance metric. The computation of mean is still done in the same way as for standard k-means. Method `SKMeans` is used to compute `k` clusters for an input, based on cosine distances.

### Requirements

Requires `scipy` and `numpy` to execute. To install, follow the instruction on [this page](https://www.scipy.org/install.html).

### Usage

There is only one method `SKMeans` in the source file `skmeans.py`. To instantiate it, 

```bash
>>> from skmeans import SKMeans
>>> no_clusters = 300
>>> kmeans_inst = SKMeans(no_clusters,iters=15)
```

Here, `no_clusters` are the number of clustes to be generated and `iters` is the number of iterations for which k-means will be run. The default value of `iters` is set to 300.

To run k-means on an input matrix `X`,

```bash
>>> X = numpy.random.rand((150,50))
>>> kmeans_inst.fit(X)
```

`X` can be a sparse matrix or a numpy array. In this case, when no keyword arguments are provided, the centres are sampled randomly from `X`. To provide, your own centre values,

```bash
>>> centres = numpy.random.rand((10,50))
>>> kmeans_inst.fit(X,sample=False,param_centres=centres)
``` 

Here `centres` can be a sparse matrix or a numpy 2d array. The default value of `param_centres` is `None`. If `sample` is set to `False`, `param_centres` should be prvided a matrix.

In case, two pass k-means is to be used (In two pass kmeans, in the first pass, a small set of the input matrix is used to sample centres, then k-means is run over this small set of input and the centres. The new centres found from the first pass are then passed to the second pass of k-means, with the complete input matrix.), call `fit` method with the following parameter,

```bash
>>> kmeans_inst.fit(X,two_pass=True)
```

The default value of the flag `two_pass` is `False`. Setting it to `True`, it will take precedence over the `sample` flag and ignore it's value. The number of input samples for the first pass of k-means is determined by the following condition,

```bash
>>> no_samples = max(2*np.sqrt(X.shape[0]), 10*self.no_clusters)
```

### Distance Calculation

`SKMeans` uses a matrix multiplication to calculate the cosine distances, hence, it is fairly fast in partice as compared to other methods, which use `scipy.spatial.distance.cdist` to compute the cosine distances.

### Note

This work is free. You can redistribute it and/or modify it under the terms of the Do Whatever You Want To Public License.