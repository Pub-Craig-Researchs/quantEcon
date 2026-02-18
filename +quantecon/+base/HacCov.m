function [hac, lags] = HacCov(u, opts)
%HACCOV Heteroskedasticity and Autocorrelation Consistent (HAC) Covariance
%
%   Implementation of Newey-West (1987) with Bartlett kernel.
%
%   Usage:
%       [hac, lags] = quantecon.base.HacCov(u, 'Lags', 4);
%
%   Inputs:
%       u - (T x N) Residuals from a regression
%
%   Options:
%       'Lags' - (int) Truncation lag. If 0 (default), uses floor(4*(T/100)^(2/9)).
%       'Prewhite' - (bool) Perform VAR(1) prewhitening (Default: false).

arguments
    u (:,:) double
    opts.Lags (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.Prewhite (1,1) logical = false
end

[T, N] = size(u);

if opts.Prewhite
    % VAR(1) Prewhitening
    Y = u(2:end, :);
    X = u(1:end-1, :);
    beta = (X' * X) \ (X' * Y);
    v = Y - X * beta;
    [T_eff, ~] = size(v);
else
    v = u;
    T_eff = T;
    beta = zeros(N, N);
end

% Select Lags if not provided
if opts.Lags == 0
    lags = floor(4 * (T_eff / 100)^(2/9));
else
    lags = opts.Lags;
end

% Sample Covariance (reduced form)
Gamma0 = (v' * v) / T_eff;
vcv = Gamma0;

% Sum of autocovariances with Bartlett kernel weights
for j = 1:lags
    weight = 1 - j / (lags + 1);
    Gamma_j = (v(j+1:end, :)' * v(1:end-j, :)) / T_eff;
    vcv = vcv + weight * (Gamma_j + Gamma_j');
end

% Recolor if prewhitened
if opts.Prewhite
    D = inv(eye(N) - beta');
    hac = D * vcv * D';
else
    hac = vcv;
end
end
