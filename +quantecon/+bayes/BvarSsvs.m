function results = BvarSsvs(Y, lags, opts)
%BVARSSVS Bayesian VAR with Stochastic Search Variable Selection
%
%   Algorithm: Spike and Slab prior for VAR coefficients to identify sparsity.
%   Reference: George, Sun, and Ni (2008).

arguments
    Y (:,:) double
    lags (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.nsamp (1,1) double = 1000
    opts.burnin (1,1) double = 500
    opts.tau0 (1,1) double = 0.1  % Spike standard deviation
    opts.tau1 (1,1) double = 10   % Slab standard deviation
end

rng(1, "twister");
[T, N] = size(Y);
T_eff = T - lags;
Y_eff = Y(lags+1:end, :);

% Build Regressors X (Pre-allocated)
X = ones(T_eff, 1 + N*lags);
for l = 1:lags
    X(:, (l-1)*N+2 : l*N+1) = Y(lags+1-l : end-l, :);
end
k = size(X, 2);
q = N * k;

% Initial Values
Beta = (X' * X) \ (X' * Y_eff); % OLS start
beta = Beta(:);
Gamma = ones(q, 1); % Indicators (1 = Slab, 0 = Spike)
Sigma = eye(N);
invSigma = inv(Sigma);

% Hyperparams
tau0 = opts.tau0;
tau1 = opts.tau1;
prob_prior = 0.5; % Prior probability for slab

% Records
beta_rec = zeros(q, opts.nsamp - opts.burnin);
gamma_rec = zeros(q, opts.nsamp - opts.burnin);
sigma_rec = zeros(N*N, opts.nsamp - opts.burnin);

for i = 1:opts.nsamp
    % 1. Draw Gamma (Indicators)
    for j = 1:q
        d0 = normpdf(beta(j), 0, tau0);
        d1 = normpdf(beta(j), 0, tau1);
        p1 = (d1 * prob_prior) / (d1 * prob_prior + d0 * (1 - prob_prior) + 1e-12);
        Gamma(j) = rand < p1;
    end

    % 2. Draw Beta (Conditional on Gamma and Sigma)
    V_prior_diag = zeros(q, 1);
    V_prior_diag(Gamma == 1) = tau1^2;
    V_prior_diag(Gamma == 0) = tau0^2;
    invV_prior = diag(1./V_prior_diag);

    % Posterior Precision
    % For memory efficiency, avoid full kron(invSigma, eye(T))
    Prec = invV_prior + kron(invSigma, X' * X);

    L = chol(Prec + eye(q)*1e-9, 'lower');

    % Mean RHS: vec(X' * Y_eff * invSigma)
    rhs = vec(X' * Y_eff * invSigma);

    beta = L' \ (L \ rhs + randn(q, 1));

    % 3. Draw Sigma (IW)
    resid = Y_eff - X * reshape(beta, k, N);
    Sbar = resid' * resid + eye(N)*0.1;
    Sigma = wishrnd(inv(Sbar), T_eff + N + 1);
    invSigma = inv(Sigma);

    if i > opts.burnin
        idx = i - opts.burnin;
        beta_rec(:, idx) = beta;
        gamma_rec(:, idx) = Gamma;
        sigma_rec(:, idx) = Sigma(:);
    end
end

results.Coefficients = reshape(mean(beta_rec, 2), k, N);
results.Probabilities = reshape(mean(gamma_rec, 2), k, N);
results.Sigma = reshape(mean(sigma_rec, 2), N, N);
end

function v = vec(M)
v = M(:);
end
