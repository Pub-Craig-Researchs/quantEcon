function results = RobustSearch(Y, opts)
%ROBUSTSEARCH Forward Search Algorithm for Outlier Detection
%
%   Algorithm: Automatically builds a "clean" subset of data
%   to identify masks for contamination. Inspired by FSDA.

arguments
    Y (:,:) double
    opts.m0 (:,:) double = [] % Initial subset size (allow empty for default)
end

[n, p] = size(Y);
if isempty(opts.m0)
    m0 = p + 1;
else
    m0 = opts.m0;
end

% Start with a clean subset
% Simplified: assume first m0 are clean for the forward process base
subset = 1:m0;
monitoring = zeros(n - m0 + 1, 1);

for m = m0 : n
    Y_sub = Y(subset, :);
    mu = mean(Y_sub, 1);
    Sigma = cov(Y_sub);

    if det(Sigma) <= 0
        Sigma = Sigma + 1e-6 * eye(p);
    end

    d2 = zeros(n, 1);
    for i = 1:n
        diff = Y(i, :) - mu;
        d2(i) = diff / Sigma * diff';
    end

    [d2_sorted, idx] = sort(d2);

    if m < n
        monitoring(m - m0 + 1) = d2_sorted(m + 1);
        subset = idx(1 : m + 1);
    end
end

results.Distances = d2;
results.Outliers = find(d2 > chi2inv(0.975, p));
results.Monitoring = monitoring;
results.CleanSubset = subset;
end
