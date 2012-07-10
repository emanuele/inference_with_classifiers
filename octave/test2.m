#!/usr/bin/octave -qf

PRNI2012_CMs = prni2012_confusion_matrices();

Xs = {PRNI2012_CMs.pred_all PRNI2012_CMs.pred_group PRNI2012_CMs.pred_random};

addpath('.:SetPartFolder')

for X_cell = Xs
    X = X_cell{1}
    partitions = SetPartition(size(X,1));

    alpha = ones(size(X));

    % uniform prior on hypotheses: p(H_i)
    prior_H = ones(1,length(partitions)) / length(partitions);

    for i = linspace(1,length(partitions),length(partitions))
        logp_X_given_H(i) = partial_independence(X, partitions{i}, alpha);
    end

    [logp_X_given_H_sorted, IX] = sort(logp_X_given_H,'descend');

    % normalization constant: p(X)
    logp_X = logp_X_given_H(1) + log(prior_H(1));
    for i = linspace(2,length(partitions),length(partitions)-1)
        logp_X = logaddexp(logp_X, logp_X_given_H(i) + log(prior_H(i)));
    end

    % p(H|X) from Bayes rule:
    log_posterior_H_given_X = logp_X_given_H + log(prior_H) - logp_X;

    for i = linspace(1,length(partitions),length(partitions))
        if exist('part_name','var')
            clear('part_name')
        end
        for j = linspace(1,length(partitions{IX(i)}),length(partitions{IX(i)}))
            part_name{j} = strcat('[',num2str(partitions{IX(i)}{j}),']');
        end
        name_2_disp = '';
        for j = linspace(1,length(part_name),length(part_name))
            name_2_disp = strcat(name_2_disp,part_name{j});
        end
        fprintf('Partition: [%s], logp_H: %.8e\n', name_2_disp, logp_X_given_H_sorted(i))
        fprintf('Partition: [%s], postP: %.8e\n', name_2_disp, exp(log_posterior_H_given_X(IX(i))))
    end

end
