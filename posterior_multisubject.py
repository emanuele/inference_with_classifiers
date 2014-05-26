import numpy as np
from partial_independence import compute_logp_H
from partitioner import Partition

def compute_log_posteriors_multisubject(Xs, partitions=None, alpha=None, prior_H=None, verbose=False):
    """Compute log of p(H|Xs) for all the Hs, i.e. the partitions, of
    the classes, given the list of confusion matrices Xs for different
    (independent subjects), the Dirichlet prior alpha and the
    hypotheses' prior p(H) (prior_H).
    """
    if partitions is None:
        partitions = list(Partition(range(Xs[0].shape[0])))

    if alpha is None:
        if verbose: print "Assuming non-informative Dirichlet prior."
        alpha = np.ones(Xs[0].shape)

    if prior_H is None:
        if verbose: print "Assuming uniform prior for p(H_i)."
        prior_H = np.ones(len(partitions)) / len(partitions)

    logp_X_given_H = np.zeros((len(partitions), len(Xs)))
    for i, partition in enumerate(partitions):
        for j in range(len(Xs)):
            logp_X_given_H[i,j] = compute_logp_H(Xs[j], partition, alpha=alpha)

    # normalization constant: p(X)
    logp_X = reduce(np.logaddexp, logp_X_given_H.sum(1) + np.log(prior_H))
    # p(H|X) from Bayes rule:
    log_posterior_H_given_X = logp_X_given_H.sum(1) + np.log(prior_H) - logp_X
    return log_posterior_H_given_X, partitions
    

    
if __name__ == '__main__':

    print "Compute the posterior probability over all hypotheses/partitions from the confusion matrices of multiple subjects."

    Xs = np.array([[[ 7, 3],
                    [ 3, 7]],
                   [[ 8, 2],
                    [ 2, 8]],
                   [[ 6, 4],
                    [ 4, 6]]])
    
    print "Xs:"
    print Xs
    partitions = list(Partition(range(Xs[0].shape[0])))
    alpha = np.ones(Xs[0].shape) # uniform prior on confusion matrices
    print "alpha:"
    print alpha

    # uniform prior on hypotheses: p(H_i)
    prior_H = np.ones(len(partitions)) / len(partitions)

    log_posterior_H_given_Xs, partitions = compute_log_posteriors_multisubject(Xs, partitions, alpha, prior_H)
    
    idx = np.argsort(log_posterior_H_given_Xs)[::-1]

    print
    for k, i in enumerate(idx[:5]):
        print "%s) p(%s | X) = %s" % (k+1, partitions[i], np.exp(log_posterior_H_given_Xs[i]))
