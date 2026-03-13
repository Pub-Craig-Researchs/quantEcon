function results = Bekk(Y, opts)
%BEKK BEKK-GARCH(1,1) Estimation
%
%   Usage:
%       results = quantecon.multivariate.Bekk(Y)
%
%   Inputs:
%       Y - (T x N) Matrix of returns
%
%   Options:
%       'Type' - 'Scalar', 'Diagonal', or 'Full' (Default: 'Scalar')
%
%   Outputs:
%       results - Struct with estimated matrices C, A, B and conditional covariances.

arguments
    Y (:,:) double {mustBeNumeric, mustBeReal}
    opts.Type (1,1) string {mustBeMember(opts.Type, ["Scalar", "Diagonal", "Full"])} = "Scalar"
end

[T, N] = size(Y);

% Starting Values
C0 = vech(chol(cov(Y))');
if opts.Type == "Scalar"
    theta0 = [C0; 0.1; 0.8];
elseif opts.Type == "Diagonal"
    theta0 = [C0; 0.1 * ones(N, 1); 0.8 * ones(N, 1)];
else % Full
    theta0 = [C0; 0.1 * vec(eye(N)); 0.8 * vec(eye(N))];
end

% Optimization
options_opt = optimset('Display', 'off', 'Algorithm', 'interior-point');
obj = @(p) -bekk_likelihood(p, Y, opts.Type);

[theta_hat, max_ll] = fmincon(obj, theta0, [], [], [], [], [], [], [], options_opt);

% Extract Covariances
[~, H] = bekk_likelihood(theta_hat, Y, opts.Type);

results.Parameters = theta_hat;
results.LogLikelihood = -max_ll;
results.Covariances = H; % (N x N x T)
results.nObs = T;
end

function [ll, H] = bekk_likelihood(theta, Y, type)
[T, N] = size(Y);
num_cov = N * (N + 1) / 2;

% Parse
C_vec = theta(1:num_cov);
C = ivech(C_vec);
CC = C * C';

idx = num_cov;
if type == "Scalar"
    A = theta(idx+1) * eye(N);
    B = theta(idx+2) * eye(N);
elseif type == "Diagonal"
    A = diag(theta(idx+1:idx+N));
    B = diag(theta(idx+N+1:idx+2*N));
else
    A = reshape(theta(idx+1:idx+N^2), N, N);
    B = reshape(theta(idx+N^2+1:idx+2*N^2), N, N);
end

% Recursion
H = zeros(N, N, T);
H(:, :, 1) = cov(Y);
ll = 0;

for t = 2:T
    eps_prev = Y(t-1, :)';
    H(:, :, t) = CC + A' * (eps_prev * eps_prev') * A + B' * H(:, :, t-1) * B;

    % Likelihood
    Ht = H(:, :, t);
    % Ensure positive definiteness for inversion
    if det(Ht) <= 0, Ht = Ht + 1e-6 * eye(N); end

    ll = ll - 0.5 * (N * log(2 * pi) + log(det(Ht)) + Y(t, :) * pinv(Ht) * Y(t, :)');
end
end

function v = vech(M)
v = M(tril(true(size(M))));
end

function M = ivech(v)
n = round((-1 + sqrt(1 + 8 * length(v))) / 2);
M = zeros(n);
M(tril(true(n))) = v;
end
