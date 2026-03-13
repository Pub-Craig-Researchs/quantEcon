function results = DfmBlock(X, opts)
%DFMBLOCK Block-Restricted Dynamic Factor Model via EM-Kalman Filter
%
%   Estimates a block-restricted DFM with idiosyncratic dynamics:
%       X_t = Lambda * F_t + v_t        (observation equation)
%       F_t = A1*F_{t-1} + ... + Ap*F_{t-p} + u_t  (factor transition)
%       v_t = D1*v_{t-1} + ... + Ds*v_{t-s} + e_t  (idio dynamics)
%   Block restrictions confine loadings so that variable groups load only
%   on designated factor blocks. Built on Banbura & Modugno (2010).
%
%   Usage:
%       results = DfmBlock(X, 'Blocks', blocks, 'NFactors', [1 1])
%
%   Inputs:
%       X  - (T x N) double, panel of stationary series (NaN = missing)
%
%   Options (Name-Value):
%       Blocks      - (N x nBlocks) logical/double, block membership matrix
%                     blocks(i,b)=1 means variable i loads on block b
%                     (default: ones(N,1), single block)
%       NFactors    - (1 x nBlocks) int, factors per block (default: 1 per block)
%       NLags       - int, factor VAR lag order (default: 2)
%       NIdioLags   - int, idiosyncratic AR lag order (default: 1)
%       MaxIter     - int, max EM iterations (default: 500)
%       Tol         - double, convergence threshold (default: 1e-3)
%       Standardize - logical (default: true)
%
%   Output:
%       results - struct with fields:
%           .States    - (T x nS) smoothed full state vector
%           .Factors   - (T x sum(r*p)) factor states
%           .Xhat      - (T x N) fitted data (original scale)
%           .C         - (N x nS) observation loading matrix
%           .A         - (nS x nS) transition matrix
%           .Q         - (nS x nS) state innovation covariance
%           .R         - (N x N) diagonal measurement noise covariance
%           .LogLik    - scalar, final log-likelihood
%           .NIter     - scalar, iterations used
%
%   Reference:
%       Banbura & Modugno (2010); Miranda-Agrippino (2014)
%
%   See also: quantecon.factor.Dfm

arguments
    X (:,:) double
    opts.Blocks double = []
    opts.NFactors double = []
    opts.NLags (1,1) double {mustBePositive, mustBeInteger} = 2
    opts.NIdioLags (1,1) double {mustBePositive, mustBeInteger} = 1
    opts.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 500
    opts.Tol (1,1) double {mustBePositive} = 1e-3
    opts.Standardize (1,1) logical = true
end

[T, N] = size(X);

% Defaults
blocks = opts.Blocks;
if isempty(blocks); blocks = ones(N, 1); end
nB = size(blocks, 2);

r = opts.NFactors;
if isempty(r); r = ones(1, nB); end

p = opts.NLags;
s = opts.NIdioLags;

% Standardize
if opts.Standardize
    mX = mean(X, 'omitnan'); vX = std(X, 0, 1, 'omitnan');
    xNaN = bsxfun(@minus, X, mX);
    xNaN = bsxfun(@rdivide, xNaN, vX);
else
    mX = zeros(1, N); vX = ones(1, N);
    xNaN = X;
end

% State dimensions
nSf  = sum(r .* p);       % factor states
nSiM = N * s;             % idio states
nS   = nSf + nSiM;

% Fill NaN for initialization (spline)
y_est = dfmb_fill_nan(xNaN);

% Initialize
SSinit = dfmb_init(y_est, blocks, r, p, s, N, nB, nSf, nSiM, nS);

% EM iterations
llOld = -inf;
SS = SSinit;

nIter = 0;
for it = 1:opts.MaxIter
    [SS, ll] = dfmb_em_step(y_est, SS, blocks, r, p, s, N, nB, nSf, nSiM, nS);

    delta = abs(ll - llOld);
    avg   = (abs(ll) + abs(llOld) + eps) / 2;
    if delta / avg < opts.Tol && it > 1
        nIter = it;
        break
    end
    llOld = ll;
    nIter = it;
end

% Final KF + KS on data with NaN
[KFout] = dfmb_kf(xNaN, SS);
[S_sm]  = dfmb_ks(KFout);

% Output
States = S_sm(:, 2:end)';
Xhat   = bsxfun(@plus, ...
    bsxfun(@times, States * SS.C', kron(vX, ones(T, 1))), ...
    kron(mX, ones(T, 1)));

results.States  = States;
results.Factors = States(:, 1:nSf);
results.Xhat    = Xhat;
results.C       = SS.C;
results.A       = SS.A;
results.Q       = SS.Q;
results.R       = SS.R;
results.LogLik  = ll;
results.NIter   = nIter;

end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function xFilled = dfmb_fill_nan(x)
%DFMB_FILL_NAN Fill NaN via spline interpolation + MA
[T, N] = size(x);
k = 3;

% Remove leading/ending all-NaN rows
indNaN = isnan(x);
rem1 = (sum(indNaN, 2) > N * 0.8);
nanLead = (cumsum(rem1) == (1:T)');
nanEnd  = (cumsum(rem1(end:-1:1)) == (1:T)');
nanEnd  = nanEnd(end:-1:1);
nanLE   = (nanLead | nanEnd);
xFilled = x;
xFilled(nanLE, :) = [];

indNaN2 = isnan(xFilled); %#ok<NASGU>
T2 = size(xFilled, 1); %#ok<NASGU>

for i = 1:N
    xi = xFilled(:, i);
    isn = isnan(xi);
    if ~any(isn); continue; end
    t1 = find(~isn, 1, 'first');
    t2 = find(~isn, 1, 'last');
    if isempty(t1); continue; end
    obs = find(~isn);
    xi(t1:t2) = spline(obs, xi(obs), (t1:t2)');
    isn2 = isnan(xi);
    if any(isn2)
        xi(isn2) = median(xi, 'omitnan');
    end
    pad = [xi(1)*ones(k,1); xi; xi(end)*ones(k,1)];
    xma = filter(ones(2*k+1,1)/(2*k+1), 1, pad);
    xma = xma(2*k+1:end);
    still_nan = isnan(xi);
    xi(still_nan) = xma(still_nan);
    xFilled(:, i) = xi;
end

% If rows were removed, pad back
if any(nanLE)
    xFull = x;
    xFull(~nanLE, :) = xFilled;
    % Fill removed rows with 0
    xFull(nanLE, :) = 0;
    xFilled = xFull;
end
end

function SS = dfmb_init(x, blocks, r, p, s, N, nB, nSf, nSiM, nS)
%DFMB_INIT Initialize via block PCA + LS
[T, ~] = size(x);

C_init = zeros(N, nS);
A_init = zeros(nS, nS);
Q_init = zeros(nS, nS);

nSfi = cumsum([1, r.*p]);
tempX = x;
tempF = NaN(T, nSf);

for b = 1:nB
    rb = r(b);
    bvars = find(blocks(:, b));

    vXb = cov(tempX(:, bvars));
    [V, ~] = eigs(vXb, rb);
    F = tempX(:, bvars) * V;
    F = F(:, 1:rb);

    % Loadings via regression
    tC = F \ tempX(:, bvars);
    C_init(bvars, nSfi(b):nSfi(b)+rb-1) = tC';

    % Store factor for orthogonalization
    nlF = 1;
    lagF = NaN(T - nlF, rb*(nlF+1));
    for j = 1:nlF+1
        lagF(:, rb*(j-1)+1:rb*j) = F((nlF+1)-j+1:end-(j-1), :);
    end
    tempF(:, nSfi(b):nSfi(b)+rb*p-1) = [zeros(T-size(lagF,1), rb*p); ...
        lagF(:, 1:min(rb*p, size(lagF,2))), ...
        zeros(size(lagF,1), max(0, rb*p - size(lagF,2)))];

    % Factor VAR
    lagFS = NaN(T-(p+1), rb*(p+1));
    for j = 1:p+1
        lagFS(:, rb*(j-1)+1:rb*j) = F((p+1)-j+1:end-j, :);
    end
    tA = lagFS(:, rb+1:end) \ lagFS(:, 1:rb);
    u  = lagFS(:, 1:rb) - lagFS(:, rb+1:end) * tA;
    tQ = cov(u);

    blockA = zeros(rb*p, rb*p);
    blockA(1:rb, 1:rb*p) = tA';
    if p > 1
        blockA(rb+1:end, 1:end-rb) = eye(rb*(p-1));
    end
    A_init(nSfi(b):nSfi(b+1)-1, nSfi(b):nSfi(b+1)-1) = blockA;

    blockQ = zeros(rb*p, rb*p);
    blockQ(1:rb, 1:rb) = tQ;
    Q_init(nSfi(b):nSfi(b+1)-1, nSfi(b):nSfi(b+1)-1) = blockQ;

    % Orthogonalize
    tempX = tempX - tempF(:, nSfi(b):nSfi(b)+rb-1) * C_init(:, nSfi(b):nSfi(b)+rb-1)';
end

% Idiosyncratic loadings (identity on first lag)
C_init(1:N, nSf+1:nSf+nSiM) = kron(eye(N), [1, zeros(1, s-1)]);

% Idiosyncratic AR
E = tempX;
for i = 1:N
    iE = E(:, i);
    lagE = NaN(T - s, s + 1);
    for j = 1:s+1
        lagE(:, j) = iE((s+1)-j+1:end-(j-1));
    end
    tA_i = lagE(:, 2:end) \ lagE(:, 1);
    uM   = lagE(:, 1) - lagE(:, 2:end) * tA_i;
    tQ_i = var(uM);

    blockA_i = zeros(s, s);
    blockA_i(1, :) = tA_i';
    if s > 1; blockA_i(2:end, 1:end-1) = eye(s-1); end
    A_init(nSf+1+(i-1)*s:nSf+i*s, nSf+1+(i-1)*s:nSf+i*s) = blockA_i;

    blockQ_i = zeros(s, s);
    blockQ_i(1, 1) = tQ_i;
    Q_init(nSf+1+(i-1)*s:nSf+i*s, nSf+1+(i-1)*s:nSf+i*s) = blockQ_i;
end

R_init = diag(1e-4 * ones(N, 1));

S0 = zeros(nS, 1);
vecP = (eye(nS^2) - kron(A_init, A_init)) \ Q_init(:);
P0 = reshape(vecP, nS, nS);

SS.S = S0; SS.P = P0;
SS.C = C_init; SS.R = R_init;
SS.A = A_init; SS.Q = Q_init;
end

function [SS, ll] = dfmb_em_step(x, SS, blocks, r, p, s, N, nB, nSf, nSiM, nS)
%DFMB_EM_STEP One EM iteration

[KFout] = dfmb_kf(x, SS);
[S_sm, P_sm, PP_sm] = dfmb_ks(KFout);
ll = KFout.ll;

[T, ~] = size(x);

% Sufficient statistics
P_sum   = zeros(nS);
Pl_sum  = zeros(nS);
PPl_sum = zeros(nS);
for t = 1:T
    P_sum   = P_sum   + P_sm{t+1};
    Pl_sum  = Pl_sum  + P_sm{t};
    PPl_sum = PPl_sum + PP_sm{t+1};
end

% Factor transition update (per block)
A_end = zeros(nS);
Q_end = zeros(nS);
C_end = zeros(N, nS);

nSfi = cumsum([1, r.*p]);

for b = 1:nB
    rb = r(b);
    idx = nSfi(b):nSfi(b+1)-1;

    E_FF  = S_sm(idx, 2:end) * S_sm(idx, 2:end)' + P_sum(idx, idx);
    E_FlFl = S_sm(idx, 1:end-1) * S_sm(idx, 1:end-1)' + Pl_sum(idx, idx);
    E_FFl  = S_sm(idx, 2:end) * S_sm(idx, 1:end-1)' + PPl_sum(idx, idx);

    tA = E_FFl / E_FlFl;
    blockA = zeros(rb*p);
    blockA(1:rb, :) = tA(1:rb, :);
    if p > 1; blockA(rb+1:end, 1:end-rb) = eye(rb*(p-1)); end
    A_end(idx, idx) = blockA;

    tQ = (1/T) * (E_FF - tA * E_FFl');
    blockQ = zeros(rb*p);
    blockQ(1:rb, 1:rb) = tQ(1:rb, 1:rb);
    Q_end(idx, idx) = blockQ;
end

% Idiosyncratic transition
for i = 1:N
    idx = nSf+1+(i-1)*s : nSf+i*s;
    E_ee  = S_sm(idx, 2:end) * S_sm(idx, 2:end)' + P_sum(idx, idx);
    E_elel = S_sm(idx, 1:end-1) * S_sm(idx, 1:end-1)' + Pl_sum(idx, idx);
    E_eel  = S_sm(idx, 2:end) * S_sm(idx, 1:end-1)' + PPl_sum(idx, idx);

    tA = E_eel / E_elel;
    blockA = zeros(s);
    blockA(1, :) = tA(1, :);
    if s > 1; blockA(2:end, 1:end-1) = eye(s-1); end
    A_end(idx, idx) = blockA;

    tQ = (1/T) * (E_ee - tA * E_eel');
    blockQ = zeros(s);
    blockQ(1, 1) = tQ(1, 1);
    Q_end(idx, idx) = blockQ;
end

% Observation loadings (per super-block)
superB = unique(blocks, 'rows');
nSB = size(superB, 1);

MloadPerSB = zeros(nSB, nSf);
nSfi2 = cumsum([1, r.*p]);
for b = 1:nB
    MloadPerSB(:, nSfi2(b):nSfi2(b)+r(b)-1) = repmat(superB(:, b), 1, r(b));
end
MloadPerSB = logical(MloadPerSB);

iMload = zeros(N, nS);
iMload(1:N, nSf+1:nSf+nSiM) = kron(eye(N), [1, zeros(1, s-1)]);
iMload = logical(iMload);

for sb = 1:nSB
    selectV = find(ismember(blocks, superB(sb,:), 'rows'));
    nV = numel(selectV);
    nFL = sum(MloadPerSB(sb, :));

    MC1 = zeros(nV * nFL);
    MC2 = zeros(nV, nFL);

    for t = 1:T
        Wx = x(t, selectV)';
        W  = diag(~isnan(Wx));
        Wx(isnan(Wx)) = 0;

        fIdx = MloadPerSB(sb, :);
        sfst = S_sm(fIdx, t+1);

        MC1 = MC1 + kron(sfst * sfst' + P_sm{t+1}(fIdx, fIdx), W);

        iIdx = any(iMload(selectV, :), 1);
        MC2 = MC2 + Wx * sfst' - ...
            W * (S_sm(iIdx, t+1) * sfst' + P_sm{t+1}(iIdx, fIdx));
    end

    if rcond(MC1) > 1e-15
        vecC = MC1 \ MC2(:);
    else
        vecC = pinv(MC1) * MC2(:);
    end
    tC = reshape(vecC, nV, nFL);
    C_end(selectV, MloadPerSB(sb, :)) = tC;
end

% Idiosyncratic loadings fixed
C_end(1:N, nSf+1:nSf+nSiM) = kron(eye(N), [1, zeros(1, s-1)]);

R_end = diag(1e-4 * ones(N, 1));

% Update P
vecP = (eye(nS^2) - kron(A_end, A_end)) \ Q_end(:);
P_end = reshape(vecP, nS, nS);

SS.S = S_sm(:, 1);
SS.P = P_end;
SS.C = C_end;
SS.R = R_end;
SS.A = A_end;
SS.Q = Q_end;
end

function KFout = dfmb_kf(x, SS)
%DFMB_KF Kalman filter with NaN handling

S_in = SS.S; P_in = SS.P;
C = SS.C; R = SS.R; A = SS.A; Q = SS.Q;

[T, ~] = size(x);
nS = length(S_in);

S_filt = NaN(nS, T+1); S_filt(:, 1) = S_in;
P_filt = cell(T+1, 1); P_filt{1} = P_in;
S_fore = NaN(nS, T);
P_fore = cell(T, 1);
K_store = cell(T, 1);
J_store = cell(T, 1);

ll = 0;

for t = 1:T
    S_fore(:, t) = A * S_filt(:, t);
    Pt = A * P_filt{t} * A' + Q;
    Pt = 0.5 * (Pt + Pt');
    P_fore{t} = Pt;

    keepr = ~isnan(x(t, :))';
    xt = x(t, keepr)';
    Ct = C(keepr, :);
    Rt = R(keepr, keepr);

    B = Ct * Pt;
    H = Ct * Pt * Ct' + Rt;

    K_store{t} = B' / H;
    J_store{t} = P_filt{t} * A' * pinv(Pt);

    innov = xt - Ct * S_fore(:, t);
    ll = ll + 0.5 * (log(det(H \ eye(size(H)))) - innov' / H * innov);

    S_filt(:, t+1) = S_fore(:, t) + K_store{t} * innov;
    Pnew = Pt - K_store{t} * B;
    P_filt{t+1} = 0.5 * (Pnew + Pnew');
end

KFout.S_filt = S_filt;
KFout.P_filt = P_filt;
KFout.S_fore = S_fore;
KFout.P_fore = P_fore;
KFout.K = K_store;
KFout.J = J_store;
KFout.ll = ll;
KFout.C = C;
KFout.A = A;
% Store the last time step's C for smoother PP_smooth initialization
keepr_last = ~isnan(x(T, :))';
KFout.Clast = C(keepr_last, :);
end

function [S_sm, P_sm, PP_sm] = dfmb_ks(KFout)
%DFMB_KS Kalman smoother

S_filt = KFout.S_filt;
P_filt = KFout.P_filt;
S_fore = KFout.S_fore;
P_fore = KFout.P_fore;
K = KFout.K;
J = KFout.J;
A = KFout.A;
Clast = KFout.Clast;

[nS, Tp1] = size(S_filt);
T = Tp1 - 1;

S_sm  = NaN(nS, Tp1);
P_sm  = cell(Tp1, 1);
PP_sm = cell(Tp1, 1);

S_sm(:, Tp1) = S_filt(:, Tp1);
P_sm{Tp1}    = P_filt{Tp1};
PP_sm{Tp1}   = (eye(nS) - K{T} * Clast) * A * P_filt{Tp1};

for t = T:-1:1
    S_sm(:, t) = S_filt(:, t) + J{t} * (S_sm(:, t+1) - S_fore(:, t));
    P_sm{t}    = P_filt{t} - J{t} * (P_fore{t} - P_sm{t+1}) * J{t}';

    if t > 1
        PP_sm{t} = P_filt{t} * J{t-1}' + ...
            J{t} * (PP_sm{t+1} - A * P_filt{t}) * J{t-1}';
    end
end
if isempty(PP_sm{1}); PP_sm{1} = zeros(nS); end
end
