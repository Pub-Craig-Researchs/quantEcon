function results = ReducedRankReg(y, x, rank, opts)
%REDUCEDRANKREG Reduced Rank Multivariate Regression
%
%   Algorithm: Identifies rank-constrained dependency between X and Y.
%   Reference: Anderson (1951), Izenman (1975).
%
%   Usage:
%       res = quantecon.multivariate.ReducedRankReg(y, x, 2);

arguments
    y (:,:) double % (T x s) Dependent variables
    x (:,:) double % (T x r) Independent variables
    rank (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.Weight string {mustBeMember(opts.Weight, ["identity", "inverse_cov"])} = "inverse_cov"
end

[T, s] = size(y);
[Tx, r] = size(x);
assert(T == Tx, 'Y and X must have same number of observations.');

% Centering
Y = y - mean(y, 1);
X = x - mean(x, 1);

% Sample Covariances
Syy = (Y' * Y) / T;
Sxx = (X' * X) / T;
Syx = (Y' * X) / T;
Sxy = Syx';

% Weighting Matrix G
if strcmpi(opts.Weight, "inverse_cov")
    G = inv(Syy);
else
    G = eye(s);
end

% Generalized Eigenproblem
% We solve: eig(G * Syx * inv(Sxx) * Sxy)
% This identifies the components of X that explain the most variance in Y
[V, D] = eig(sqrtm(G) * Syx * (Sxx \ Sxy) * sqrtm(G));
[eig_vals, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Select rank r_eff
r_eff = min(rank, min(r, s));
V_r = V(:, 1:r_eff);

% Reduced Rank Coefficients
% B = inv(sqrt(G)) * V_r * V_r' * sqrt(G) * Syx * inv(Sxx)
G_sqrt = sqrtm(G);
G_sqrt_inv = inv(G_sqrt);

Beta_rrr = G_sqrt_inv * (V_r * V_r') * G_sqrt * (Syx / Sxx);

% Constant
Constant = mean(y, 1)' - Beta_rrr * mean(x, 1)';

results.Beta = Beta_rrr;
results.Constant = Constant;
results.EigenValues = eig_vals;
results.Rank = r_eff;
results.Residuals = Y' - Beta_rrr * X';
results.Syy = Syy;
results.ExplainedVariance = sum(eig_vals(1:r_eff)) / sum(eig_vals);
end
