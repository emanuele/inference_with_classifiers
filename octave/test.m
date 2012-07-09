#!/usr/bin/octave -qf

load('ICANN2011_CMs.mat')

X = ICANN2011_CMs.tu;

addpath('.:SetPartFolder')
partitions = SetPartition(size(X,1));

alpha = ones(size(X));

prior_H = ones(length(partitions)) / length(partitions);

for i = linspace(1,length(partitions),length(partitions))
    logp_X_given_H(i) = partial_independence(X, partitions{i}, alpha);
    partitions{i}, logp_X_given_H(i)
end
