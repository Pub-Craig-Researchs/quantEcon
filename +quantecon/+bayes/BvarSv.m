function results = BvarSv(Y, lags, opts)
%BVARSV Bayesian VAR with Stochastic Volatility
%
%   Algorithm: Gibbs Sampling for TVP-SV VAR (simplified BEAR kernel)
%
%   Usage:
%       results = quantecon.bayes.BvarSv(Y, 1, 'nsamp', 1000);

arguments
    Y (:,:) double
    lags (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.nsamp (1,1) double = 1000
    opts.burnin (1,1) double = 500
end

[T_orig, N] = size(Y);
T = T_orig - lags;

% Prep data
Y_eff = Y(lags+1:end, :);
X = ones(T, 1 + N*lags);
for l = 1:lags
    X(:, (l-1)*N+2 : l*N+1) = Y(lags+1-l : end-l, :);
end
K = size(X, 2); % Number of regressors per equation
Q = K * N;     % Total coefficients

% Priors
beta0 = zeros(Q, 1);
Omega0_inv = eye(Q) * 0.1; % inv(Omega0)

% Initial values (OLS)
B_init = (X' * X) \ (X' * Y_eff);
beta = B_init(:);

% SV Initial values
h = zeros(T, N);
phi = 0.9 * ones(N, 1);
sig_h = 0.1 * ones(N, 1);

% Pre-allocate Gibbs records
It = opts.nsamp;
Bu = opts.burnin;
beta_gibbs = zeros(Q, It - Bu);
h_gibbs = zeros(T, N, It - Bu);

% Pre-allocate transition matrices for sampling
inv_Omega_sum = zeros(Q, Q);
Xy_sum = zeros(Q, 1);

for ii = 1:It
    inv_Omega_sum(:) = 0;
    Xy_sum(:) = 0;

    for t = 1:T
        inv_Sigma_t = diag(exp(-h(t, :)));
        Xt = kron(eye(N), X(t, :));
        inv_Omega_sum = inv_Omega_sum + Xt' * inv_Sigma_t * Xt;
        Xy_sum = Xy_sum + Xt' * inv_Sigma_t * Y_eff(t, :)';
    end

    V_beta = inv(inv_Omega_sum + Omega0_inv);
    m_beta = V_beta * (Xy_sum + Omega0_inv * beta0);
    beta = m_beta + chol(V_beta, 'lower') * randn(Q, 1);

    % Sample SV (h)
    res = Y_eff - X * reshape(beta, K, N);
    for n = 1:N
        h(:, n) = sample_log_vol(res(:, n), h(:, n), phi(n), sig_h(n));
    end

    if ii > Bu
        beta_gibbs(:, ii-Bu) = beta;
        h_gibbs(:, :, ii-Bu) = h;
    end
end

results.Beta = mean(beta_gibbs, 2);
results.BetaPosterior = beta_gibbs;
results.VolPosterior = h_gibbs;
results.MeanVol = exp(mean(h_gibbs, 3));
results.X = X;
results.Y = Y_eff;
results.lags = lags;
end

function h = sample_log_vol(eps, h_old, phi, sig)
T = length(eps);
h = h_old;
for t = 1:T
    h_cand = h(t) + 0.1 * randn();

    if t == 1
        log_prior_old = -0.5 * (h(1)^2) / (sig^2 / (1-phi^2));
        log_prior_cand = -0.5 * (h_cand^2) / (sig^2 / (1-phi^2));
    else
        log_prior_old = -0.5 * (h(t) - phi*h(t-1))^2 / sig^2;
        log_prior_cand = -0.5 * (h_cand - phi*h(t-1))^2 / sig^2;
    end

    log_lik_old = -0.5 * h(t) - 0.5 * (eps(t)^2) * exp(-h(t));
    log_lik_cand = -0.5 * h_cand - 0.5 * (eps(t)^2) * exp(-h_cand);

    if log(rand()) < (log_prior_cand + log_lik_cand) - (log_prior_old + log_lik_old)
        h(t) = h_cand;
    end
end
end
