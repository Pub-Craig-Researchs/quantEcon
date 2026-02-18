function results = Ols(y, X, opts)
%OLS Ordinary Least Squares Regression
%
%   Usage:
%       results = quantecon.base.Ols(y, X)
%       results = quantecon.base.Ols(y, X, 'Robust', true)
%
%   Inputs:
%       y - (T x 1) Dependent variable
%       X - (T x K) Independent variables
%
%   Options (Name-Value):
%       'HasConstant' - (logical) Add intercept if true (Default: true)
%       'Robust'      - (logical) Use HC1 robust standard errors (Default: false)
%
%   Outputs:
%       results - Struct with coefficients, t-stats, p-values, etc.

arguments
    y (:,1) double {mustBeNumeric, mustBeReal}
    X (:,:) double {mustBeNumeric, mustBeReal}
    opts.HasConstant (1,1) logical = true
    opts.Robust (1,1) logical = false
end

[T, K] = size(X);

if opts.HasConstant
    X = [ones(T, 1), X];
    K = K + 1;
end

% Check dimensions
if T <= K
    error("quantecon:base:Ols:RankDeficient", "Number of observations must exceed number of variables.");
end

% Estimation using QR decomposition for stability
[Q_mat, R_mat] = qr(X, 0);
beta = R_mat \ (Q_mat' * y);

yhat = X * beta;
resid = y - yhat;

% Sum of Squares
SRE = resid' * resid;
SSR = SRE; % Residual Sum of Squares
SST = sum((y - mean(y)).^2);

% Variance estimation
s2 = SSR / (T - K);
XpXinv = (R_mat' * R_mat) \ eye(K);

if opts.Robust
    % HC1 Robust Standard Errors
    meat = (X .* resid)' * (X .* resid);
    V = (T / (T - K)) * (XpXinv * meat * XpXinv);
else
    % Standard Homoskedastic Errors
    V = s2 * XpXinv;
end

se = sqrt(diag(V));
tstat = beta ./ se;
pvalue = 2 * (1 - tcdf(abs(tstat), T - K));

% R-squared
rsq = 1 - SSR / SST;
adj_rsq = 1 - (SSR / (T - K)) / (SST / (T - 1));

% Store Results
results.Coefficients = beta;
results.SE = se;
results.tStat = tstat;
results.pValue = pvalue;
results.R2 = rsq;
results.AdjR2 = adj_rsq;
results.Residuals = resid;
results.YHat = yhat;
results.Covariance = V;
results.nObs = T;
results.nVar = K;
results.s2 = s2;

% Durbin-Watson
ediff = diff(resid);
results.DW = (ediff' * ediff) / SSR;

end
