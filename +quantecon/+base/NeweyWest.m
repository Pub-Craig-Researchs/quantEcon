function [V, se] = NeweyWest(X, e, L)
%NEWEYWEST Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC) Covariance
%
%   Usage:
%       [V, se] = quantecon.base.NeweyWest(X, residuals, lags)
%
%   Inputs:
%       X - (T x K) Regressor matrix
%       e - (T x 1) Residuals
%       L - (int) Number of lags (Default: floor(T^(1/3)))
%
%   Outputs:
%       V  - (K x K) HAC Covariance matrix
%       se - (K x 1) HAC Standard Errors

arguments
    X (:,:) double {mustBeNumeric, mustBeReal}
    e (:,1) double {mustBeNumeric, mustBeReal}
    L double = []
end

[T, K] = size(X);
if isempty(L)
    L = floor(T^(1/3));
end

% Moments: g_t = X_t * e_t
g = X .* e;

% Gamma(0) - Homoskedastic/White part
XpXinv = (X' * X) \ eye(K);
S = (g' * g) / T;

% Gamma(j) - Autocorrelation part
for j = 1:L
    weight = 1 - j / (L + 1);
    Gamma_j = (g(j+1:T, :)' * g(1:T-j, :)) / T;
    S = S + weight * (Gamma_j + Gamma_j');
end

% V = T * (X'X)^-1 * S * (X'X)^-1
V = T * (XpXinv * S * XpXinv);
se = sqrt(diag(V));

end
