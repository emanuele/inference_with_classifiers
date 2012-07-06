import numpy as np
from partial_independence import compute_logp_H
from icann2011_confusion_matrices import tu
from partitioner import Partition

if __name__ == '__main__':

    X = tu
    print "X:"
    print X
    partitions = list(Partition(range(X.shape[0])))
    alpha = np.ones(X.shape) # uniform prior on confusion matrices
    print "alpha:"
    print alpha

    # uniform prior on hypotheses: p(H_i)
    prior_H = np.ones(len(partitions)) / len(partitions)
    logp_X_given_H = np.zeros(len(partitions))
    
    for i, partition in enumerate(partitions):
        logp_X_given_H[i] = compute_logp_H(X, partition, alpha=alpha)

    # normalization constant: p(X)
    logp_X = reduce(np.logaddexp, logp_X_given_H + np.log(prior_H))

    # p(H|X) from Bayes rule:
    log_posterior_H_given_X = logp_X_given_H + np.log(prior_H) - logp_X

    idx = np.argsort(log_posterior_H_given_X)[::-1]

    print
    for k, i in enumerate(idx[:5]):
        print "%s) p(%s | X) = %s" % (k+1, partitions[i], np.exp(log_posterior_H_given_X[i]))
