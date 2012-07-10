function log_sum = logaddexp(loga, logb)
%%% log(a + b) given log(a) and log(b). Equivalent to numpy.logaddexp().
%%% log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
%%% Note that np.log1p(x) is a numerically more accurate function for np.log(1 + x).
%%% See: http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction

% log_sum =loga + log(1 + exp(logb - loga))
log_sum = loga + log1p(exp(logb - loga));