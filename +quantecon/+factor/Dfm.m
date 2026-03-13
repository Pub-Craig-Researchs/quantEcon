function results = Dfm(X, opts)
%DFM Dynamic Factor Model via EM-Kalman Filter
%
%   Estimates a dynamic factor model:
%       X_t = Lambda * F_t + e_t       (observation equation)
%       F_t = A1*F_{t-1} + ... + Ap*F_{t-p} + u_t  (state equation)
%   with diagonal idiosyncratic covariance, via EM algorithm with
%   Kalman filter/smoother (Miranda-Agrippino 2013).
%
%   Usage:
%       results = Dfm(X, 'NFactors', 3, 'NLags', 2)
%
%   Inputs:
%       X  - (T x N) double, panel of stationary time series
%
%   Options (Name-Value):
%       NFactors   - int, number of factors r (default: 1)
%       NLags      - int, factor VAR lag order p (default: 1)
%       MaxIter    - int, max EM iterations (default: 1000)
%       Tol        - double, convergence threshold (default: 1e-3)
%       Standardize - logical, standardize data first (default: true)
%
%   Output:
%       results - struct with fields:
%           .Factors   - (T x r) smoothed factors
%           .States    - (T x r*p) full state vector
%           .Xhat      - (T x N) fitted/smoothed data (original scale)
%           .Lambda    - (N x r) loading matrix
%           .A         - (r*p x r*p) companion transition matrix
%           .Q         - (r*p x r*p) state innovation covariance
%           .R         - (N x N) diagonal idiosyncratic covariance
%           .LogLik    - scalar, final log-likelihood
%           .NIter     - scalar, iterations used
%
%   Reference:
%       Miranda-Agrippino (2013), EM algorithm for DFM
%
%   See also: quantecon.factor.Pca

arguments
    X (:,:) double
    opts.NFactors (1,1) double {mustBePositive, mustBeInteger} = 1
    opts.NLags (1,1) double {mustBePositive, mustBeInteger} = 1
    opts.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 1000
    opts.Tol (1,1) double {mustBePositive} = 1e-3
    opts.Standardize (1,1) logical = true
end

[T, N] = size(X);
r = opts.NFactors;
p = opts.NLags;

% Standardize
if opts.Standardize
    mX = mean(X, 'omitnan');
    vX = std(X, 0, 1, 'omitnan');
    x  = bsxfun(@minus, X, mX);
    x  = bsxfun(@rdivide, x, vX);
else
    mX = zeros(1, N);
    vX = ones(1, N);
    x  = X;
end

% Initialize via PCA + LS
[S_init, P_init, C_init, R_init, A_init, Q_init] = dfm_init(x, r, p);

% EM iterations
it = 0; %#ok<NASGU>
llOld = 0;

S_cur = S_init; P_cur = P_init;
C_cur = C_init; R_cur = R_init;
A_cur = A_init; Q_cur = Q_init;

for it = 1:opts.MaxIter
    [S_cur, P_cur, C_cur, R_cur, A_cur, Q_cur, ll] = ...
        dfm_em_step(x, r, p, S_cur, P_cur, C_cur, R_cur, A_cur, Q_cur);

    if abs(ll / (llOld + 1e-30) - 1) < opts.Tol && it > 1
        break
    end
    llOld = ll;
end

% Final Kalman filter + smoother
[S_filt, P_filt, S_fore, P_fore, KL] = ...
    dfm_kf(x, S_cur, P_cur, C_cur, R_cur, A_cur, Q_cur);

[S_sm, ~] = dfm_ks(S_filt, P_filt, S_fore, P_fore, KL, C_cur, A_cur);

% Pack output
Factors = S_sm(1:r, 2:end)';
States  = S_sm(:, 2:end)';
Xhat    = bsxfun(@plus, ...
    bsxfun(@times, Factors * C_cur(:, 1:r)', kron(vX, ones(T, 1))), ...
    kron(mX, ones(T, 1)));

results.Factors = Factors;
results.States  = States;
results.Xhat    = Xhat;
results.Lambda  = C_cur(:, 1:r);
results.A       = A_cur;
results.Q       = Q_cur;
results.R       = R_cur;
results.LogLik  = ll;
results.NIter   = it;

end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [S0, P0, C0, R0, A0, Q0] = dfm_init(x, r, p)
%DFM_INIT Initialize EM with PCA + least squares
[T, N] = size(x);
vX = cov(x);
[V, D] = eigs(vX, r);
F = x * V / sqrt(D);

% Observation equation
tempC = V * sqrt(D);
e = x - F * tempC';
tempR = cov(e);

C0 = zeros(N, r*p);
C0(:, 1:r) = tempC;
R0 = diag(diag(tempR));

% Transition equation
lagF = NaN(T - p, r*(p+1));
for j = 1:p+1
    lagF(:, r*(j-1)+1 : r*j) = F((p+1) - j + 1 : end - (j-1), :);
end
tempA = (lagF(:, r+1:end)' * lagF(:, r+1:end)) \ ...
    (lagF(:, r+1:end)' * lagF(:, 1:r));
u = lagF(:, 1:r) - lagF(:, r+1:end) * tempA;
tempQ = cov(u);

A0 = zeros(r*p, r*p);
A0(1:r, :) = tempA';
if p > 1
    A0(r+1:end, 1:r*(p-1)) = eye(r*(p-1));
end

Q0 = zeros(r*p, r*p);
Q0(1:r, 1:r) = tempQ;

S0 = zeros(r*p, 1);
vecP = (eye((r*p)^2) - kron(A0, A0)) \ Q0(:);
P0 = reshape(vecP, r*p, r*p);
end

function [S_end, P_end, C_end, R_end, A_end, Q_end, ll] = ...
    dfm_em_step(x, r, p, S_in, P_in, C_in, R_in, A_in, Q_in)
%DFM_EM_STEP One EM iteration: E-step (KF+KS), M-step (update params)

% E-step
[S_filt, P_filt, S_fore, P_fore, KL, ~, ~, ~, ~, ll] = ...
    dfm_kf(x, S_in, P_in, C_in, R_in, A_in, Q_in);

[S_sm, P_sm, PP_sm] = dfm_ks(S_filt, P_filt, S_fore, P_fore, KL, C_in, A_in);

% M-step
[T, N] = size(x);

P_sum   = zeros(r*p, r*p);
Pl_sum  = zeros(r*p, r*p);
PPl_sum = zeros(r*p, r*p);

for t = 1:T
    P_sum   = P_sum   + P_sm{t+1};
    Pl_sum  = Pl_sum  + P_sm{t};
    PPl_sum = PPl_sum + PP_sm{t+1};
end

% Sufficient statistics using full state vector for proper VAR(p)
E_FF   = S_sm(1:r, 2:end) * S_sm(1:r, 2:end)' + P_sum(1:r, 1:r);
E_SlSl = S_sm(:, 1:end-1) * S_sm(:, 1:end-1)' + Pl_sum;
E_FSl  = S_sm(1:r, 2:end) * S_sm(:, 1:end-1)' + PPl_sum(1:r, :);

% Transition: A(1:r,:) = E[F_t * S_{t-1}'] / E[S_{t-1} * S_{t-1}']
tempA = E_FSl / (E_SlSl + 1e-10*eye(r*p));
tempQ = (1/T) * (E_FF - tempA * E_FSl');
tempQ = 0.5*(tempQ + tempQ');  % symmetrize

% Observation loadings: C = E[X_t * F_t'] / E[F_t * F_t']
tempC = (x' * S_sm(1:r, 2:end)') / (E_FF + 1e-10*eye(r));
ee    = (x' - tempC * S_sm(1:r, 2:end)) * (x' - tempC * S_sm(1:r, 2:end))';
tempR = (1/T) * (ee + tempC * P_sum(1:r, 1:r) * tempC');

A_end = zeros(r*p, r*p);
A_end(1:r, :) = tempA;  % full row: [A1, A2, ..., Ap]
if p > 1
    A_end(r+1:end, 1:r*(p-1)) = eye(r*(p-1));
end

Q_end = zeros(r*p, r*p);
Q_end(1:r, 1:r) = tempQ;
Q_end(1:r, 1:r) = max(Q_end(1:r,1:r), 0);  % ensure non-negative diagonal

C_end = zeros(N, r*p);
C_end(:, 1:r) = tempC;

R_end = diag(diag(tempR));

S_end = S_sm(:, 2);
P_end = P_sum;
end

function [S_filt, P_filt, S_fore, P_fore, KL, C, R, A, Q, ll] = ...
    dfm_kf(x, S_in, P_in, C, R, A, Q)
%DFM_KF Kalman filter

T  = size(x, 1);
nS = length(S_in);

S_filt = NaN(nS, T+1); S_filt(:, 1) = S_in;
P_filt = cell(T+1, 1); P_filt{1} = P_in;
S_fore = NaN(nS, T);   P_fore = cell(T, 1);
KL     = cell(T, 2);

ll = 0;

for t = 1:T
    S_fore(:, t) = A * S_filt(:, t);
    P_fore{t}    = A * P_filt{t} * A' + Q;

    Pt = 0.5*(P_fore{t} + P_fore{t}');  % symmetrize
    P_fore{t} = Pt;

    B = C * Pt;
    H = C * Pt * C' + R;
    H = 0.5*(H + H') + 1e-10*eye(size(H,1));  % regularize

    KL{t, 1} = B' / H;
    KL{t, 2} = P_filt{t} * A' / (Pt + 1e-10*eye(size(Pt,1)));

    innov = x(t, :)' - C * S_fore(:, t);
    [cholH, pflag] = chol(H, 'lower');
    if pflag == 0
        logdetH = 2*sum(log(diag(cholH)));
    else
        logdetH = log(max(det(H), realmin));
    end
    ll = ll - 0.5 * (logdetH + innov' * (H \ innov));

    S_filt(:, t+1) = S_fore(:, t) + KL{t, 1} * innov;
    Pnew = Pt - KL{t, 1} * B;
    P_filt{t+1}    = 0.5*(Pnew + Pnew');  % symmetrize
end
end

function [S_sm, P_sm, PP_sm] = dfm_ks(S_filt, P_filt, S_fore, P_fore, KL, C, A)
%DFM_KS Kalman smoother

[nS, Tp1] = size(S_filt);
T = Tp1 - 1;

S_sm  = NaN(nS, Tp1);
P_sm  = cell(Tp1, 1);
PP_sm = cell(Tp1, 1);

K = KL(:, 1);
L = KL(:, 2);

S_sm(:, Tp1)  = S_filt(:, Tp1);
P_sm{Tp1}     = P_filt{Tp1};
PP_sm{Tp1}    = (eye(nS) - K{T} * C) * A * P_filt{Tp1};

for t = T:-1:1
    S_sm(:, t) = S_filt(:, t) + L{t} * (S_sm(:, t+1) - S_fore(:, t));
    P_sm{t}    = P_filt{t} - L{t} * (P_fore{t} - P_sm{t+1}) * L{t}';

    if t > 1
        PP_sm{t} = P_filt{t} * L{t-1}' + ...
            L{t} * (PP_sm{t+1} - A * P_filt{t}) * L{t-1}';
    end
end

% PP_sm{1} not set, initialize
if isempty(PP_sm{1})
    PP_sm{1} = zeros(nS);
end
end
