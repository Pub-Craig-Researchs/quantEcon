function results = ClusterReg(y, X, G, opts)
%CLUSTERREG OLS with Multi-Way Clustered Standard Errors
%
%   Usage:
%       results = quantecon.panel.ClusterReg(y, X, G)
%
%   Inputs:
%       y - (T x 1) Dependent variable
%       X - (T x K) Independent variables
%       G - (T x M) Group indices for clustering (up to 2-way)
%
%   Options:
%       'HasConstant' - (logical) Add intercept if true (Default: true)
%
%   Outputs:
%       results - Struct with coefficients, clustered SEs, and statistics.

arguments
    y (:,1) double {mustBeNumeric, mustBeReal}
    X (:,:) double {mustBeNumeric, mustBeReal}
    G (:,:) double {mustBeNumeric, mustBeReal}
    opts.HasConstant (1,1) logical = true
end

[T, K] = size(X);
[Tg, M] = size(G);

if T ~= Tg
    error("quantecon:panel:ClusterReg:DimensionMismatch", "Group indices must have same length as observations.");
end

if opts.HasConstant
    X = [ones(T, 1), X];
    K = K + 1;
end

% OLS Estimation
[Q_mat, R_mat] = qr(X, 0);
beta = R_mat \ (Q_mat' * y);
resid = y - X * beta;

% Bread of the Sandwich: (X'X)^-1
XpXinv = (R_mat' * R_mat) \ eye(K);

% Meat: Omega
if M == 1
    Omega = compute_meat(X, resid, G(:,1));
elseif M == 2
    Omega1 = compute_meat(X, resid, G(:,1));
    Omega2 = compute_meat(X, resid, G(:,2));
    [~, ~, G12] = unique(G, "rows");
    Omega12 = compute_meat(X, resid, G12);
    Omega = Omega1 + Omega2 - Omega12;
else
    error("quantecon:panel:ClusterReg:Unsupported", "Currently only support up to 2-way clustering.");
end

V = XpXinv * Omega * XpXinv;

se = sqrt(diag(V));
tstat = beta ./ se;
pvalue = 2 * (1 - normcdf(abs(tstat)));

SSR = resid' * resid;
SST = sum((y - mean(y)).^2);
rsq = 1 - SSR / SST;

results.Coefficients = beta;
results.SE = se;
results.tStat = tstat;
results.pValue = pvalue;
results.Covariance = V;
results.R2 = rsq;
results.nObs = T;
results.nGroups = M;
results.Residuals = resid;

end

function Omega = compute_meat(X, resid, g)
[T, K] = size(X);
[ids, ~, idx] = unique(g);
nGroups = length(ids);
Scores = X .* resid;
GroupScores = zeros(nGroups, K);
for k = 1:K
    GroupScores(:, k) = accumarray(idx, Scores(:, k));
end
Omega = GroupScores' * GroupScores;
c = (nGroups / (nGroups - 1)) * ((T - 1) / (T - K));
Omega = c * Omega;
end
