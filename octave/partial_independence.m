function logp_H = partial_independence(X, psi, alpha)
%%% Partial independence model: one margin fixed (rows margin).
%%% Analytical solution

    logp_H = compute_logp_H(X, psi, alpha);
end

function logp_ib = compute_logp_independent_block(X, alpha)
%%% Compute the analytical log likelihood of a matrix under the assumption 
%%% of independence between rows and columns.

    logp_ib = gammaln(sum(alpha(:))) - sum(gammaln(alpha(:)));
    term1_2_sum = gammaln(ones(size(alpha,1),1)*sum(X,1) + alpha);
    logp_ib = logp_ib + sum(term1_2_sum(:)) - gammaln(sum(X(:)) + sum(alpha(:)));
    term2_2_sum = gammaln(sum(X,2) + 1);
    term3_2_sum = gammaln(X + 1);
    logp_ib = logp_ib + sum(term2_2_sum(:)) - sum(term3_2_sum(:));

end

function logp_H = compute_logp_H(X, psi, alpha)
%%% Compute the analytical log likelihood of the confusion matrix X
%%% with hyper-prior alpha (in a multivariate-Dirichlet sense)
%%% according to the partitioning scheme psi

    logp_H = 0.0;
    for group = psi
        if length(group{1}) == 1
            logp_H = logp_H + log_multivariate_polya(X(group{1},:), alpha(group{1},:));
            
        else
            % which classes are not involved
            nogroup = setdiff(linspace(1,size(X,1),size(X,1)),group{1});
            for i = group{1}
                logp_H = logp_H + log_multivariate_polya([sum(X(i,group{1})) X(i,nogroup(:))],[sum(alpha(i,group{1})) alpha(i,nogroup(:))] );           
            end
            logp_H = logp_H + compute_logp_independent_block(X(group{1},group{1}),sum(alpha(group{1},group{1}),2)');
            
        end
    end
end