function result = CommodityFutures(LogFutures, Maturities, opts)
% CommodityFutures  Multi-factor commodity futures pricing via Kalman filter MLE
%
%   Estimates a 4-factor state-space model for commodity futures prices where
%   the latent state vector is x = [log S, delta, r, V]:
%     (1) log spot price
%     (2) convenience yield
%     (3) interest rate
%     (4) stochastic volatility of the log-price
%
%   Futures prices follow a quasi-affine representation:
%     log F(t,T) = a(T-t) + Z(T-t) * x_t + epsilon_t
%   where a(.) and Z(.) are affine coefficients computed via 4th-order
%   Runge-Kutta integration.  Parameters are estimated by maximum likelihood
%   using the Kalman filter prediction error decomposition, initialized from
%   randomized multi-start optimization.
%
%   Two estimation modes (Ballestra & Tezza, Quantitative Finance):
%     (A) Commodity-only: all 4 factors latent, inferred from futures prices
%     (B) With rates: interest rate observed from bond yields, appended as
%         an extra observation equation  r_obs = [0,0,1,0] * x_t + eta_t
%
% SYNTAX:
%   % Mode A: commodity-only
%   result = quantecon.finance.CommodityFutures(LogFutures, Maturities)
%
%   % Mode B: with observed interest rates
%   result = quantecon.finance.CommodityFutures(LogFutures, Maturities, ...
%               'Rates', r_observed)
%
%   % Full options
%   result = quantecon.finance.CommodityFutures(LogFutures, Maturities, ...
%               'Rates', r_obs, 'TimeStep', 1/252, 'NRunge', 2000, ...
%               'NStarts', 100)
%
% INPUTS:
%   LogFutures  - (T x ny) matrix of log futures prices, one column per contract
%   Maturities  - (1 x ny) vector of contract maturities in years
%
% OPTIONS (name-value):
%   Rates       - (T x 1) observed interest rates / bond yields (default: [])
%                 When provided, adds r_t as an observable in the Kalman filter.
%   TimeStep    - observation time step in years (default: 1/252 for daily)
%   NRunge      - Runge-Kutta discretization steps (default: 2000)
%   NStarts     - number of random starting points (default: 100)
%   Display     - fmincon display option: 'off'|'iter'|'final' (default: 'off')
%   Algorithm   - fmincon algorithm: 'sqp'|'interior-point' (default: 'sqp')
%
% OUTPUT:
%   result struct with fields:
%     .Theta      - (1 x nparam) estimated parameter vector
%     .LogLik     - maximized log-likelihood
%     .AIC        - Akaike Information Criterion
%     .BIC        - Bayesian Information Criterion
%     .States     - (T x 4) filtered state estimates [logS, delta, r, V]
%     .Residuals  - (T x n_obs) prediction errors (ny futures + 1 rate if Rates given)
%     .AffineA    - (1 x ny) affine intercepts a(T_i) (futures only)
%     .AffineZ    - (ny x 4) affine loadings Z(T_i) (futures only)
%     .ParamTable - table of named parameter estimates
%     .EigMin     - (T x 1) minimum eigenvalue of F_t (diagnostics)
%     .T          - number of time observations
%     .ny         - number of futures contracts
%     .HasRates   - logical, whether interest rate observations were used
%
% PARAMETER VECTOR (27 + n_obs elements, n_obs = ny or ny+1):
%   1  mu1hat     risk-neutral mean log spot price
%   2  mu2        physical mean convenience yield
%   3  mu2hat     risk-neutral mean convenience yield
%   4  mu3        physical mean interest rate
%   5  mu3hat     risk-neutral mean interest rate
%   6  mu4        physical mean volatility
%   7  mu4hat     risk-neutral mean volatility
%   8  k2         physical mean-reversion speed (convenience yield)
%   9  k2hat      risk-neutral mean-reversion speed (convenience yield)
%  10  k3         physical mean-reversion speed (interest rate)
%  11  k3hat      risk-neutral mean-reversion speed (interest rate)
%  12  k4         physical mean-reversion speed (volatility)
%  13  k4hat      risk-neutral mean-reversion speed (volatility)
%  14-18           s12, s22, s13, s23, s33   (Omega_0 elements)
%  19-24           rho12..rho34              (correlation parameters)
%  25-27           sig22, sig33, sig44       (volatility diffusion scales)
%  28-(27+ny)      sigepsi_1 .. sigepsi_ny   (futures measurement error std devs)
%  (28+ny)         sigeta                    (rate measurement error, if Rates given)
%
% REFERENCE:
%   Ballestra, L.V. and Tezza, C. "A Multi-Factor Model for Improved
%   Commodity Pricing: Calibration and an Application to the Oil Market",
%   Quantitative Finance.
%
% Source: Refactored from SRV_four_factors-main (Tezza & Ballestra).

arguments
    LogFutures (:,:) double
    Maturities (1,:) double {mustBePositive}
    opts.Rates     (:,1) double = double.empty(0,1)
    opts.TimeStep  (1,1) double {mustBePositive} = 1/252
    opts.NRunge    (1,1) double {mustBePositive, mustBeInteger} = 2000
    opts.NStarts   (1,1) double {mustBePositive, mustBeInteger} = 100
    opts.Display   (1,:) char = 'off'
    opts.Algorithm (1,:) char {mustBeMember(opts.Algorithm, ...
        {'sqp','interior-point'})} = 'sqp'
end

% --- Input validation ---
[T, ny] = size(LogFutures);
hasRates = ~isempty(opts.Rates);

assert(numel(Maturities) == ny, ...
    'CommodityFutures:dimMismatch', ...
    'Maturities length (%d) must equal number of futures columns (%d).', ...
    numel(Maturities), ny);

assert(T > 10, ...
    'CommodityFutures:tooFew', ...
    'At least 10 observations required (got %d).', T);

assert(all(isfinite(LogFutures(:))), ...
    'CommodityFutures:nanInf', ...
    'LogFutures must not contain NaN or Inf.');

if hasRates
    assert(numel(opts.Rates) == T, ...
        'CommodityFutures:ratesDim', ...
        'Rates length (%d) must equal number of observations T (%d).', ...
        numel(opts.Rates), T);
    assert(all(isfinite(opts.Rates)), ...
        'CommodityFutures:ratesNanInf', ...
        'Rates must not contain NaN or Inf.');
end

dt     = opts.TimeStep;
nrunge = opts.NRunge;

% --- Augment observation matrix if rates observed ---
if hasRates
    yt_full = [LogFutures, opts.Rates(:)];   % (T x ny+1)
    n_obs   = ny + 1;
else
    yt_full = LogFutures;                     % (T x ny)
    n_obs   = ny;
end

% --- Phase 1: Multi-start initialization ---
nparam_fixed = 27;
nparam       = nparam_fixed + n_obs;

theta0 = generate_starts(yt_full, Maturities, nrunge, dt, opts.NStarts, ny);

% --- Phase 2: Constrained MLE via fmincon ---
fmin_opts = optimoptions('fmincon', ...
    'Display',                opts.Display, ...
    'Algorithm',              opts.Algorithm, ...
    'MaxFunctionEvaluations', 5000 * nparam, ...
    'MaxIterations',          2000);

lb = repmat(-10, 1, nparam);
ub = repmat(+10, 1, nparam);
lb(19:24) = -1;   % correlation bounds
ub(19:24) = +1;

objfun = @(th) kalman_negloglik(th, yt_full, Maturities, nrunge, dt, ny);

theta_best = theta0;
ll_best    = objfun(theta0);

% Suppress singular-matrix warnings during optimization (expected for
% some random parameter draws in the search phase)
wstate = warning('off', 'MATLAB:nearlySingularMatrix');
wstate2 = warning('off', 'MATLAB:singularMatrix');
cleanWarn = onCleanup(@() warning(wstate));
cleanWarn2 = onCleanup(@() warning(wstate2));

try
    theta_opt = fmincon(objfun, theta0, ...
        [], [], [], [], lb, ub, [], fmin_opts);
    ll_opt = objfun(theta_opt);
    if isfinite(ll_opt) && ll_opt < ll_best
        theta_best = theta_opt;
        ll_best    = ll_opt;
    end
catch ME
    warning('CommodityFutures:optFail', ...
        'fmincon failed (%s). Using best starting point.', ME.message);
end

% --- Phase 3: Extract final estimates ---
[~, ~, States, Residuals, a_coef, Z_coef, eig_min] = ...
    kalman_negloglik(theta_best, yt_full, Maturities, nrunge, dt, ny);

LogLik = -ll_best;

% Information criteria
T_obs = T * n_obs;
AIC   = -2 * LogLik + 2 * nparam;
BIC   = -2 * LogLik + log(T_obs) * nparam;

% Named parameter table
pnames     = build_param_names(ny, hasRates);
ParamTable = table(pnames(:), theta_best(:), ...
    'VariableNames', {'Parameter', 'Estimate'});

% --- Assemble output ---
result = struct( ...
    'Theta',      theta_best, ...
    'LogLik',     LogLik, ...
    'AIC',        AIC, ...
    'BIC',        BIC, ...
    'States',     States, ...
    'Residuals',  Residuals, ...
    'AffineA',    a_coef, ...
    'AffineZ',    Z_coef, ...
    'ParamTable', ParamTable, ...
    'EigMin',     eig_min, ...
    'T',          T, ...
    'ny',         ny, ...
    'HasRates',   hasRates);

end

% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================

function theta0 = generate_starts(yt, TF, nrunge, dt, nsim, nfut)
% GENERATE_STARTS  Randomized multi-start initialization.
%   Evaluates nsim random parameter vectors and returns the one with lowest
%   (real, finite) negative log-likelihood.
%   nfut = number of futures contracts (columns of TF).
%   If size(yt,2) > nfut, the extra column is an observed rate.

ny_obs = size(yt, 2);
nparam = 27 + ny_obs;
ll_vec = inf(nsim, 1);
p_mat  = zeros(nsim, nparam);

% Suppress singular-matrix warnings during random start evaluation
ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
ws2 = warning('off', 'MATLAB:singularMatrix');
cw1 = onCleanup(@() warning(ws1));
cw2 = onCleanup(@() warning(ws2));

for i = 1:nsim

    % Means
    mu1hat = rand;
    mu2 = rand;  mu2hat = rand;
    mu3 = rand;  mu3hat = rand;
    mu4 = rand;  mu4hat = rand;

    % Mean-reversion speeds
    k2 = rand;  k2hat = rand;
    k3 = rand;  k3hat = rand;
    k4 = rand;  k4hat = rand;

    % Omega_0 elements
    s12 = 0.05 * rand;
    s22 = 0.05 * rand;
    s13 = 0.05 * rand;
    s23 = 0.05 * rand;
    s33 = 0.05 * rand;

    % Correlations
    rho12 = 0.8 * rand;
    rho13 = 0.8 * rand - 0.8;
    rho14 = 0.8 * rand - 0.5;
    rho23 = 0.5 * rand - 0.5;
    rho24 = 0.5 * rand - 0.5;
    rho34 = 0.5 * rand - 0.5;

    % Volatility diffusion scales
    sig22 = 0.1 * sqrt(rand);
    sig33 = 0.1 * sqrt(rand);
    sig44 = 0.1 * sqrt(rand);

    % Measurement error std devs (futures)
    epsi = 0.01 * rand(1, nfut);

    % Rate measurement error (if applicable)
    if ny_obs > nfut
        sigeta = 0.005 * rand;
        epsi = [epsi, sigeta]; %#ok<AGROW>
    end

    theta = [mu1hat, mu2, mu2hat, mu3, mu3hat, mu4, mu4hat, ...
             k2, k2hat, k3, k3hat, k4, k4hat, ...
             s12, s22, s13, s23, s33, ...
             rho12, rho13, rho14, rho23, rho24, rho34, ...
             sig22, sig33, sig44, ...
             epsi];

    val = kalman_negloglik(theta, yt, TF, nrunge, dt, nfut);

    % [FIX]: Original used min(ll(imag(ll)==0)) which could match multiple
    % rows.  We track per-iteration and select unique best via min index.
    if isreal(val) && isfinite(val)
        ll_vec(i) = val;
    end
    p_mat(i, :) = theta;
end

[~, idx] = min(ll_vec);
theta0 = p_mat(idx(1), :);   % idx(1) guarantees single selection
end

% ---------------------------------------------------------------------

function [ll, lp, Xt, et, a_coef, Z_coef, eig_val] = ...
        kalman_negloglik(theta, yt, TrungeF, nrunge, h, nfut)
% KALMAN_NEGLOGLIK  Negative log-likelihood via Kalman filter.
%   Returns total negative log-likelihood (scalar for fmincon) plus
%   filtered states, residuals, and affine coefficients.
%   nfut  = number of futures contracts (for RK affine coefficients).
%   If size(yt,2) > nfut, the extra column(s) are observed rates.

n      = size(yt, 1);   % number of observations
ny_obs = size(yt, 2);   % total observation dimension (nfut or nfut+1)
d      = 4;             % state dimension

% --- Unpack parameters ---
mu1hat = theta(1);
mu2    = theta(2);   mu2hat = theta(3);
mu3    = theta(4);   mu3hat = theta(5);
mu4    = theta(6);   mu4hat = theta(7);
k2     = theta(8);   k2hat  = theta(9);
k3     = theta(10);  k3hat  = theta(11);
k4     = theta(12);  k4hat  = theta(13);
s12    = theta(14);  s22    = theta(15);
s13    = theta(16);  s23    = theta(17);  s33 = theta(18);
rho12  = theta(19);  rho13  = theta(20);  rho14 = theta(21);
rho23  = theta(22);  rho24  = theta(23);  rho34 = theta(24);
sig22  = theta(25);  sig33  = theta(26);  sig44 = theta(27);
sigepsi = theta(28:27+ny_obs);

% --- Measurement error covariance ---
H = diag(sigepsi.^2);

% --- Physical measure drift ---
A = [0; mu2*k2; mu3*k3; mu4*k4];

B = [0, -1, +1, -0.5;
     0, -k2,  0,    0;
     0,   0, -k3,   0;
     0,   0,   0,  -k4];

% --- Risk-neutral drift ---
Af = [mu1hat; mu2hat*k2hat; mu3hat*k3hat; mu4hat*k4hat];

% [NOTE]: Original code sets Bf = B (physical drift matrix used for state
% prediction).  This is a deliberate modeling choice by Ballestra & Tezza
% that equates the physical and risk-neutral B matrix in the Kalman
% prediction step, while retaining separate Af vs A intercepts.
Bf = B;

% --- Covariance structure ---
% Omega_0: constant component
omega0 = [0,    s12,    s13,    0;
          s12,  s22^2,  s23,    0;
          s13,  s23,    s33^2,  0;
          0,    0,      0,      0];

% Omega_1: volatility-dependent component (scaled by V_t)
omega1 = [1,            rho12*sig22,       rho13*sig33,       rho14*sig44;
          rho12*sig22,  sig22^2,           rho23*sig22*sig33, rho24*sig22*sig44;
          rho13*sig33,  rho23*sig22*sig33, sig33^2,           rho34*sig33*sig44;
          rho14*sig44,  rho24*sig22*sig44, rho34*sig33*sig44, sig44^2];

% --- Runge-Kutta: affine coefficients for each maturity ---
nmat   = numel(TrungeF);
a_coef = zeros(1, nmat);
Z_coef = zeros(nmat, d);

for j = 1:nmat
    [a_coef(j), Z_coef(j, :)] = ...
        runge_kutta_affine(nrunge, TrungeF(j), A, B, omega0, omega1);
end

% --- Augment observation model if rate data present ---
if ny_obs > nfut
    % Append rate observation:  r_obs = 0 + [0,0,1,0] * x_t + eta_t
    a_obs = [a_coef, 0];                      % (1 x ny_obs)
    Z_obs = [Z_coef; 0, 0, 1, 0];             % (ny_obs x 4)
else
    a_obs = a_coef;
    Z_obs = Z_coef;
end

% --- Kalman filter ---
I_d = eye(d);
Phi = I_d + h * Bf;            % precomputed transition matrix

ll  = 0;
lp  = zeros(n, 1);
Xt  = zeros(n, d);
et  = zeros(n, ny_obs);
eig_val = zeros(n, 1);

% Initial state: risk-neutral long-run means
Xt0 = [mu1hat, mu2hat, mu3hat, mu4hat];

% Initial state covariance
Pt     = eye(d);
Pt(4,4) = 0.001;              % tighter initial variance for volatility

for i = 1:n
    if i > 1
        Xt0 = Xt(i-1, :);
    end

    % --- Prediction ---
    Xpred    = h * Af' + (Phi * Xt0')';

    % Exponential discretization for the volatility factor
    Xpred(4) = exp(-h * k4hat) * (Xt0(4) + k4hat * mu4hat * h);

    ypred = a_obs + Xpred * Z_obs';     % (1 x ny_obs)

    Q     = omega0 + omega1 * Xt0(4);     % state-dependent covariance
    Ppred = Phi * Pt * Phi' + h * Q;

    % --- Innovation ---
    vt = yt(i, :) - ypred;
    Ft = Z_obs * Ppred * Z_obs' + H;

    % Eigenvalue check for invertibility diagnostics
    eig_val(i) = min(eig(Ft));

    % Guard: if Ft is singular or non-PD, return large penalty
    if eig_val(i) < 1e-14
        ll = 1e10;
        return
    end

    % --- Update ---
    Kt       = (Ppred * Z_obs') / Ft;
    Xt(i, :) = Xpred + (Kt * vt')';
    Pt       = Ppred - Kt * Z_obs * Ppred;

    % --- Log-likelihood contribution (neg-loglik, positive) ---
    lp(i) = 0.5 * (ny_obs * log(2*pi) + log(det(Ft)) + vt * (Ft \ vt'));
    ll     = ll + lp(i);

    et(i, :) = vt;
end

end

% ---------------------------------------------------------------------

function [a_final, z_final] = runge_kutta_affine(nsteps, T_mat, A, B, omega0, omega1)
% RUNGE_KUTTA_AFFINE  4th-order Runge-Kutta for quasi-affine coefficients.
%   Solves the coupled ODE system for the scalar a(tau) and the 1x4 loading
%   vector Z(tau) that appear in the log-futures price formula.
%
%   ODE system (autonomous in tau):
%     da/dtau = Z * A + 0.5 * Z * Omega_0 * Z'
%     dZ/dtau = [Z*B(:,1), Z*B(:,2), Z*B(:,3), Z*B(:,4) + 0.5*Z*Omega_1*Z']
%
%   Initial conditions:  a(0) = 0,  Z(0) = [1, 0, 0, 0].

h_rk = T_mat / nsteps;

% ODE right-hand sides (autonomous: t-argument unused)
da = @(b) b * A + 0.5 * (b * omega0 * b');
db = @(b) [b*B(:,1), b*B(:,2), b*B(:,3), b*B(:,4) + 0.5*(b*omega1*b')];

% Initial conditions
a = 0;
b = [1, 0, 0, 0];

for step = 1:nsteps
    % [FIX]: Original code incorrectly used scalar da/dtau (ka1) instead of
    % vector db/dtau (kb1) when computing the intermediate b-state for the
    % a-equation stages.  Since da/dtau = f(b) and db/dtau = g(b), both
    % intermediate evaluations must use the same b-state increment (kb).
    % With nrunge=2000 the error is small, but mathematically incorrect.

    kb1 = db(b);
    ka1 = da(b);

    b_mid1 = b + h_rk/2 * kb1;
    kb2 = db(b_mid1);
    ka2 = da(b_mid1);

    b_mid2 = b + h_rk/2 * kb2;
    kb3 = db(b_mid2);
    ka3 = da(b_mid2);

    b_end = b + h_rk * kb3;
    kb4 = db(b_end);
    ka4 = da(b_end);

    a = a + h_rk/6 * (ka1 + 2*ka2 + 2*ka3 + ka4);
    b = b + h_rk/6 * (kb1 + 2*kb2 + 2*kb3 + kb4);
end

a_final = a;
z_final = b;
end

% ---------------------------------------------------------------------

function names = build_param_names(ny, hasRates)
% BUILD_PARAM_NAMES  Named parameter labels for the parameter vector.
%   ny = number of futures contracts.  If hasRates, append 'sigeta'.

names = { ...
    'mu1hat', 'mu2', 'mu2hat', 'mu3', 'mu3hat', 'mu4', 'mu4hat', ...
    'k2', 'k2hat', 'k3', 'k3hat', 'k4', 'k4hat', ...
    's12', 's22', 's13', 's23', 's33', ...
    'rho12', 'rho13', 'rho14', 'rho23', 'rho24', 'rho34', ...
    'sig22', 'sig33', 'sig44'};

for j = 1:ny
    names{end+1} = sprintf('sigepsi_%d', j); %#ok<AGROW>
end

if hasRates
    names{end+1} = 'sigeta';
end
end
