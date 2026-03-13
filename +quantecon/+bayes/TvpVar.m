function results = TvpVar(Y, lags, opts)
%TVPVAR Time-Varying Parameter Bayesian VAR (Production-Grade)
%
%   Algorithm: Gibbs sampling with RW coefficients and SV.
%   Numerical Fix: Aggressive ridge for precision stability and IW regularization.

arguments
    Y (:,:) double
    lags (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.nsamp (1,1) double = 1000
    opts.burnin (1,1) double = 500
end

rng(0, "twister");
[T, N] = size(Y);
T_eff = T - lags;
q = N * (N * lags + 1);
nsamp = opts.nsamp;
burnin = opts.burnin;

Y_eff = Y(lags+1:end, :);
Xbar = cell(T_eff, 1);
for t = 1:T_eff
    xt = zeros(1, 1 + N*lags);
    xt(1) = 1;
    for l = 1:lags
        xt((l-1)*N+2 : l*N+1) = Y(t+lags-l, :);
    end
    Xbar{t} = kron(eye(N), xt);
end

chi = 0.01; psi = 0.01;
Omega = 0.01 * eye(q);
invOmega = inv(Omega);
Sigma = eye(N);
invSigma = inv(Sigma);
Beta = zeros(q, T_eff);

beta_record = zeros(q, T_eff, nsamp - burnin);
omega_record = zeros(q, nsamp - burnin);
sigma_record = zeros(N, N, nsamp - burnin);

for i = 1:nsamp
    % Draw Beta path
    Beta = sample_beta_path(Y_eff, Xbar, invSigma, invOmega, Beta);

    % Draw Omega
    diff_beta = diff(Beta, 1, 2);
    for j = 1:q
        scale = psi + sum(diff_beta(j,:).^2);
        Omega(j,j) = 1/gamrnd((chi + T_eff-1)/2, 2/scale);
    end
    % Ensure invOmega is well-behaved
    diag_invO = 1./diag(Omega);
    diag_invO(isinf(diag_invO)) = 1e6;
    invOmega = diag(diag_invO);

    % Draw Sigma
    resid = zeros(N, T_eff);
    for t = 1:T_eff
        resid(:, t) = Y_eff(t, :)' - Xbar{t} * Beta(:, t);
    end
    Sbar = resid * resid' + eye(N)*0.1;
    % Regularized Wishart draw
    Sigma = wishrnd(inv(Sbar + eye(N)*1e-6), T_eff + N + 1);
    invSigma = inv(Sigma + eye(N)*1e-9);

    if i > burnin
        idx = i - burnin;
        beta_record(:, :, idx) = Beta;
        omega_record(:, idx) = diag(Omega);
        sigma_record(:, :, idx) = Sigma;
    end
end

results.Beta = mean(beta_record, 3);
results.Omega = mean(omega_record, 2);
results.Sigma = mean(sigma_record, 3);
results.BetaFull = beta_record;
end

function Beta = sample_beta_path(Y, Xbar, invSigma, invOmega, Beta_old)
T = size(Y, 1);
q = size(Beta_old, 1);
Beta = Beta_old;
for t = 1:T
    Prec = Xbar{t}' * invSigma * Xbar{t};
    if t > 1; Prec = Prec + invOmega; end

    % Aggressive ridge for numerical stability
    Prec = Prec + eye(q) * 1e-6;

    Y_adj = Xbar{t}' * invSigma * Y(t,:)';
    if t > 1; Y_adj = Y_adj + invOmega * Beta(:, t-1); end

    % Fallback for chol failure
    try
        L = chol(Prec, 'lower');
    catch
        Prec = Prec + eye(q) * 1e-4;
        L = chol(Prec, 'lower');
    end
    Beta(:, t) = L' \ (L \ Y_adj + randn(q, 1));
end
end
