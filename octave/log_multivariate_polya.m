function log_likelihood = log_multivariate_polya(x, alpha)
%%% PDF of the multivariate Pólya distribution.
%%% See: http://en.wikipedia.org/wiki/Multivariate_P%C3%B3lya_distribution

    x = x(:);
    alpha = alpha(:);
    size(x,1);
    size(alpha,1);
    assert(size(x,1) == size(alpha,1));
    N = sum(x);
    A = sum(alpha);
    log_likelihood = gammaln(N+1) - sum(gammaln(x+1));
    log_likelihood = log_likelihood + (gammaln(A) - sum(gammaln(alpha)));
    log_likelihood = log_likelihood + (sum(gammaln(x + alpha)) - gammaln(N + A));
    
end