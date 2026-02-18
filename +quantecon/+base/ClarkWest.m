function results = ClarkWest(actual, forecast1, forecast2)
%CLARKWEST Clark and West (2007) test for predictive accuracy (Nested Models)
%
%   Usage:
%       results = quantecon.base.ClarkWest(actual, forecast_restricted, forecast_unrestricted)
%
%   Inputs:
%       actual    - (T x 1) Actual values
%       forecast1 - (T x 1) Forecasts from the restricted (smaller) model
%       forecast2 - (T x 1) Forecasts from the unrestricted (larger) model
%
%   Outputs:
%       results - Struct with MSPE-adjusted statistic and p-value.
%
%   Reference:
%       Clark, T. E., & West, K. D. (2007). Approximately normal tests for
%       equal predictive accuracy in nested models. Journal of Econometrics.

arguments
    actual (:,1) double {mustBeNumeric, mustBeReal}
    forecast1 (:,1) double {mustBeNumeric, mustBeReal}
    forecast2 (:,1) double {mustBeNumeric, mustBeReal}
end

T = length(actual);
e1 = actual - forecast1;
e2 = actual - forecast2;

% The core of the CW test is the adjusted loss differential:
% f_t = e1_t^2 - (e2_t^2 - (y1_t - y2_t)^2)
f = e1.^2 - (e2.^2 - (forecast1 - forecast2).^2);

% Regression of f on a constant to get mean and SE
[Q, R] = qr(ones(T, 1), 0);
f_bar = R \ (Q' * f);
resid = f - f_bar;

% Newey-West Standard Error for the Constant
L = floor(4 * (T/100)^(2/9));
gamma0 = (resid' * resid) / T;
V = gamma0;
for j = 1:L
    gamma_j = (resid(j+1:T)' * resid(1:T-j)) / T;
    weight = 1 - j / (L + 1);
    V = V + 2 * weight * gamma_j;
end

cw_stat = f_bar / sqrt(V / T);
% CW test is usually one-sided (Unrestricted model is better)
p_value = 1 - normcdf(cw_stat);

results.Statistic = cw_stat;
results.pValue = p_value;
results.MeanDifference = f_bar;
results.nObs = T;
end
