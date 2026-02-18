function results = GmmFit(moment_func, theta0, Y, X, Z, opts)
%GMMFIT Generalized Method of Moments (GMM) Estimation
%
%   Usage:
%       results = quantecon.base.GmmFit(moment_func, theta0, Y, X, Z)
%
%   Inputs:
%       moment_func - Function handle: @(theta, type, Y, X, Z, W)
%                     type 1: returns objective (scalar)
%                     type 2: returns moment matrix (T x K)
%                     type 3: returns mean moments (K x 1)
%       theta0      - (q x 1) Initial parameter values
%       Y           - (T x N) Dependent variables
%       X           - (T x L) Independent variables
%       Z           - (T x K) Instruments
%
%   Options:
%       'MaxIter'   - (int) Maximum GMM iterations (Default: 10)
%       'Tol'       - (double) Convergence tolerance (Default: 1e-4)
%       'HacLag'    - (int) Lag for Newey-West HAC (Default: automatic)
%
%   Outputs:
%       results - Struct with estimated parameters, t-stats, etc.

arguments
    moment_func function_handle
    theta0 (:,1) double {mustBeNumeric, mustBeReal}
    Y (:,:) double {mustBeNumeric, mustBeReal}
    X (:,:) double {mustBeNumeric, mustBeReal}
    Z (:,:) double {mustBeNumeric, mustBeReal}
    opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 10
    opts.Tol (1,1) double {mustBePositive} = 1e-4
    opts.HacLag = []
end

T = size(Y, 1);
K = size(Z, 2); % Number of moment conditions
q = length(theta0); % Number of parameters

if isempty(opts.HacLag)
    opts.HacLag = floor(T^(1/3));
end

% 1-Step GMM (Identity Weighting)
W = eye(K);
options_opt = optimset("Display", "off", "LargeScale", "off", "MaxFunEvals", 10000);

theta = theta0;
fv_prev = inf;

for i = 1:opts.MaxIter
    % Optimize given W
    obj = @(p) moment_func(p, 1, Y, X, Z, W);
    [theta, fv] = fminsearch(obj, theta, options_opt);

    % Check Convergence
    if abs(fv - fv_prev) / (abs(fv_prev) + eps) < opts.Tol
        break;
    end
    fv_prev = fv;

    % Update Weighting Matrix using HAC
    moments = moment_func(theta, 2, Y, X, Z, W);
    W = compute_hac_weight(moments, opts.HacLag);
end

% Inference
mean_moments = moment_func(theta, 3, Y, X, Z, W);

% Jacobian Matrix (Numerical)
G = zeros(K, q);
for j = 1:q
    delta = max(theta(j) * 1e-4, 1e-6);
    theta_plus = theta;
    theta_plus(j) = theta_plus(j) + delta;
    G(:, j) = (moment_func(theta_plus, 3, Y, X, Z, W) - mean_moments) / delta;
end

% Covariance: V = (1/T) * (G'W G)^-1 G'W S W G (G'W G)^-1
% Where W = S^-1 at the optimum
V = pinv(G' * W * G) / T;
se = sqrt(diag(V));
tstat = theta ./ se;
pvalue = 2 * (1 - normcdf(abs(tstat)));

% J-test (Overidentifying restrictions)
J_stat = T * fv;
J_pvalue = 1 - chi2cdf(J_stat, K - q);

% Results
results.Parameters = theta;
results.SE = se;
results.tStat = tstat;
results.pValue = pvalue;
results.Covariance = V;
results.JStat = J_stat;
results.JPValue = J_pvalue;
results.Iterations = i;
results.nObs = T;
results.nMoments = K;
end

function W = compute_hac_weight(moments, L)
[T, ~] = size(moments);
% Center moments
moments = moments - mean(moments);

% Gamma(0)
S = (moments' * moments) / T;

% Gamma(j)
for j = 1:L
    weight = 1 - j / (L + 1);
    Gamma_j = (moments(j+1:T, :)' * moments(1:T-j, :)) / T;
    S = S + weight * (Gamma_j + Gamma_j');
end

% W is inverse of spectral density S
[U, D, V_mat] = svd(S);
d = diag(D);
d_inv = 1 ./ (d + eps);
W = V_mat * diag(d_inv) * U';
end
