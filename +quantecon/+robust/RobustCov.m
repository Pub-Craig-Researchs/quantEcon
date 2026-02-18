function results = RobustCov(Y, opts)
%ROBUSTCOV Robust Mean and Covariance using Minimum Covariance Determinant (MCD)
%
%   Usage:
%       results = quantecon.robust.RobustCov(Y)
%
%   Inputs:
%       Y - (T x N) Data matrix
%
%   Options:
%       'H' - (int) Subset size (Default: floor((T+N+1)/2))
%       'nsamp' - (int) Number of subsamples (Default: 500)
%
%   Outputs:
%       results - Struct with Mean, Cov, MD (Mahalanobis Distances), and Outliers.

arguments
    Y (:,:) double {mustBeNumeric, mustBeReal}
    opts.H (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.nsamp (1,1) double {mustBeInteger, mustBePositive} = 500
end

[T, N] = size(Y);
if opts.H == 0
    h = floor((T + N + 1) / 2);
else
    h = opts.H;
end

best_obj = Inf;
best_mean = zeros(1, N);
best_cov = zeros(N, N);

% Fast-MCD Sampling
for i = 1:opts.nsamp
    % Elemental subset of size N+1
    idx = randsample(T, N + 1);
    Yk = Y(idx, :);

    mk = mean(Yk);
    Sk = cov(Yk);

    if det(Sk) <= 0, continue; end

    % C-step
    diff = Y - mk;
    md2 = sum((diff / Sk) .* diff, 2);
    [~, sort_idx] = sort(md2);

    h_idx = sort_idx(1:h);
    Yh = Y(h_idx, :);
    m_h = mean(Yh);
    S_h = cov(Yh);
    obj_h = det(S_h);

    if obj_h < best_obj
        best_obj = obj_h;
        best_mean = m_h;
        best_cov = S_h;
    end
end

% Consistency Correction (approximate for normal distribution)
alpha = h / T;
factor = alpha / chi2cdf(chi2inv(alpha, N), N + 2);
best_cov = best_cov * factor;

% Mahalanobis Distances
diff = Y - best_mean;
md2 = sum((diff / best_cov) .* diff, 2);

% Outliers using Chi-square cut-off
cutoff = chi2inv(0.975, N);
outliers = find(md2 > cutoff);

results.Mean = best_mean;
results.Cov = best_cov;
results.MD = sqrt(md2);
results.Outliers = outliers;
results.nObs = T;
results.h = h;
end
