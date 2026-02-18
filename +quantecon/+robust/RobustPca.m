function results = RobustPca(Y, k, opts)
%ROBUSTPCA Robust Principal Component Analysis via MCD Covariance
%
%   Usage:
%       results = quantecon.robust.RobustPca(Y, k)
%
%   Inputs:
%       Y - (T x N) Data matrix
%       k - (int) Number of components to retain
%
%   Outputs:
%       results - Struct with Loadings, Scores, and Explained variance.

arguments
    Y (:,:) double {mustBeNumeric, mustBeReal}
    k (1,1) double {mustBeInteger, mustBePositive}
    opts.H (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.nsamp (1,1) double {mustBeInteger, mustBePositive} = 500
end

% Use RobustCov to get robust covariance matrix
rob_cov = quantecon.robust.RobustCov(Y, 'H', opts.H, 'nsamp', opts.nsamp);

% Eigendecomposition of robust covariance
[V, D] = eig(rob_cov.Cov);
[eig_vals, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Select k components
loadings = V(:, 1:k);
explained = eig_vals(1:k) / sum(eig_vals);

% Centering based on robust mean
Y_centered = Y - rob_cov.Mean;
scores = Y_centered * loadings;

results.Loadings = loadings;
results.Scores = scores;
results.Explained = explained;
results.RobustMean = rob_cov.Mean;
results.RobustCov = rob_cov.Cov;
results.Outliers = rob_cov.Outliers;
end
