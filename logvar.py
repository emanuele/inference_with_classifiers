"""Numerically stable log(mean) and log(var) from log(data).
"""

import numpy as np

def logaddexp(loga, logb):
    """ log(a + b) given log(a) and log(b). Equivalent to numpy.logaddexp().

    log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))

    Note that np.log1p(x) is a numerically more accurate function for np.log(1 + x).

    See: http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    """
    # return loga + np.log(1 + np.exp(logb - loga))
    return loga + np.log1p(np.exp(logb - loga))


def logaddexp_vector(logvector, axis=-1):
    """ log(v)1 + ... + v_n) given logvector=[log(v_1),...,log(v_n)].
    """
    return np.logaddexp.reduce(logvector)


def logsubexp(loga, logb):
    """ log(a - b) given log(a) and log(b).
    
    log(a - b) = log(a) + log(1 - exp(log(b) - log(a))) (when a>b).

    Note that np.log1p(-x) is a numerically more accurate function for np.log(1 - x).

    See: http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    """
    assert((loga>logb).all())
    # return loga + np.log(1.0 - np.exp(logb - loga))
    return loga + np.log1p(-np.exp(logb - loga))

def logabssubexp(loga, logb):
    """ log(|a - b|) given log(a) and log(b). This function is a clever mix of
    log(a) + log(1 - exp(log(b) - log(a))) (when a>b) and
    log(b) + log(1 - exp(log(a) - log(b))) (when a<b), i.e.
    max(log(a),log(b)) + log(1 - exp(-abs(log(b) - log(a))))
    It is clever because works with vectors as well.
    
    Note that np.log1p(-x) is a numerically more accurate function for np.log(1 - x).

    See: http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    """
    # ALTERNATIVE VERSION: return np.max([loga, np.repeat(logb, loga.size)], axis=0) + np.log(1.0 - np.exp(-np.abs(logb - loga)))
    # return np.clip(logb, loga, logb) + np.log(1.0 - np.exp(-np.abs(logb - loga)))
    return np.clip(logb, loga, logb) + np.log1p(-np.exp(-np.abs(logb - loga)))

def logmean(loga):
    # NOTE: DOES SORTING loga IMPROVE THE ACCURACY OF THE RESULT?
    # NOTE: WHAT ABOUT KAHAN SUMMATION?
    return reduce(np.logaddexp, loga) - np.log(loga.size)

def logvar(loga):
    return - np.log(loga.size) + reduce(np.logaddexp, 2*logabssubexp(loga, logmean(loga)))


if __name__ == '__main__':

    print __doc__

    a = np.arange(1,10)
    print "data:", a
    print "On plain data:"
    print "mean, log(mean):", a.mean(), np.log(a.mean())
    print "var, log(var):", a.var(), np.log(a.var())
    print
    loga = np.log(a)
    print "log(data):", loga
    print "On logscaled data:"
    print "exp(logmean), logmean:", np.exp(logmean(loga)), logmean(loga)
    print "exp(logvar), logvar:", np.exp(logvar(loga)), logvar(loga)

    from numpy.testing import assert_array_almost_equal
    assert_array_almost_equal(a.mean(), np.exp(logmean(loga)), decimal=10)
    assert_array_almost_equal(a.var(), np.exp(logvar(loga)), decimal=10)
