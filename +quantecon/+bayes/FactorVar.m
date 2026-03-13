function results = FactorVar(XY, data_exo, nfactors, lags, opts)
%FACTORVAR Factor-Augmented Bayesian VAR (FAVAR)
%
%   Algorithm: Two-step BBE (2005) approach.
%
%   Usage:
%       results = quantecon.bayes.FactorVar(XY, [], 3, 1);

arguments
    XY (:,:) double
    data_exo (:,:) double = []
    nfactors (1,1) double {mustBeInteger, mustBePositive} = 3
    lags (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.PriorType string = "minnesota"
end

[T, M] = size(XY);

% Step 1: Principal Component Analysis
XY_std = (XY - mean(XY, 1)) ./ std(XY, 0, 1);
[V, D] = eig(XY_std' * XY_std);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Extracted factors F
F = XY_std * V(:, 1:nfactors);

% Combine Factors with Exogenous Data
if ~isempty(data_exo)
    F = [F, data_exo];
end

% Step 2: Estimate BVAR
mdl = quantecon.bayes.Bvar(lags, opts.PriorType);
res_bvar = mdl.estimate(F);

% Loadings (L)
L = (F' * F) \ (F' * XY_std);

results.Factors = F;
results.Loadings = L;
results.BvarResults = res_bvar;
results.nFactors = nfactors;
results.T = T;
results.M = M;
end
