function results = RobustReg(y, X, opts)
%ROBUSTREG Robust Regression using Least Trimmed Squares (LTS)
%
%   Usage:
%       results = quantecon.robust.RobustReg(y, X)
%
%   Inputs:
%       y - (T x 1) Response variable
%       X - (T x K) Explanatory variables
%
%   Options:
%       'H' - (int) Number of observations to trim (Default: floor((T+K+1)/2))
%       'nsamp' - (int) Number of subsamples (Default: 500)
%
%   Outputs:
%       results - Struct with Beta, Scale, Weights, and Outliers.

arguments
    y (:,1) double {mustBeNumeric, mustBeReal}
    X (:,:) double {mustBeNumeric, mustBeReal}
    opts.H (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.nsamp (1,1) double {mustBeInteger, mustBePositive} = 500
end

[T, K] = size(X);
if opts.H == 0
    h = floor((T + K + 1) / 2);
else
    h = opts.H;
end

best_obj = Inf;
best_beta = zeros(K, 1);

% Sampling
for i = 1:opts.nsamp
    % Extract elemental subset of size K
    idx = randsample(T, K);
    Xk = X(idx, :); yk = y(idx);

    % Check rank
    if rank(Xk) < K, continue; end

    % Estimate beta
    b = Xk \ yk;

    % C-step: concentration
    res2 = (y - X * b).^2;
    [~, sort_idx] = sort(res2);

    % Keep h best observations
    h_idx = sort_idx(1:h);
    b_h = X(h_idx, :) \ y(h_idx);
    obj_h = sum((y(h_idx) - X(h_idx, :) * b_h).^2);

    if obj_h < best_obj
        best_obj = obj_h;
        best_beta = b_h;
    end
end

% Final estimates
res = y - X * best_beta;
res2 = res.^2;
[~, sort_idx] = sort(res2);
weights = zeros(T, 1);
weights(sort_idx(1:h)) = 1;

% Scale estimate (consistency corrected)
scale_raw = sqrt(best_obj / (h - K));
% Asymptotic consistency factor for normal errors (approximate)
alpha = h / T;
factor = alpha / chi2cdf(chi2inv(alpha, 1), 3);
scale = scale_raw * sqrt(factor);

% Find outliers using 97.5% cut-off
cutoff = sqrt(chi2inv(0.975, 1));
std_res = res / scale;
outliers = find(abs(std_res) > cutoff);

results.Beta = best_beta;
results.Scale = scale;
results.Weights = weights;
results.Outliers = outliers;
results.Residuals = res;
results.StdResiduals = std_res;
results.nObs = T;
results.h = h;
end
