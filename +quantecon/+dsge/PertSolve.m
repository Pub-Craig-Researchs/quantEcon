function [SOL, stab] = PertSolve(A, B, nx, options)
%PERTSOLVE First-order perturbation solution for linearised DSGE models.
%
%   Solves the system of linear difference equations:
%       A * x(t+1) = B * x(t)
%
%   where x(t) = [k(t); u(t)], k(t) is the (nw x 1) vector of state
%   variables (predetermined + exogenous) and u(t) is the (ny x 1) vector
%   of control variables (jump/forward-looking).
%
%   Returns the decision rule:
%       u(t)   = F * w(t)
%       w(t+1) = P * w(t)      [where w = [x_endog; z_exog]]
%
%   INPUTS:
%       A   - (n x n) coefficient matrix on x(t+1)
%       B   - (n x n) coefficient matrix on x(t)
%       nx  - (scalar) number of predetermined endogenous state variables
%
%   Name-Value Options:
%       nz    - number of exogenous state variables.  Default: 0
%       Rho   - (nz x nz) autoregressive matrix for exogenous shocks.
%               z(t+1) = Rho * z(t) + Omega * eta(t+1). Default: []
%       Omega - (nz x nz) shock impact matrix.        Default: []
%
%   OUTPUTS:
%       SOL (struct):
%           .F     - (ny x nw) decision rule: u(t) = F * w(t)
%           .P     - (nw x nw) law of motion: w(t+1) = P * w(t)
%           .Hx_w  - (nx x nw) state part of P (first nx rows)
%           .Hy_w  - (ny x nw) same as F
%           .gev   - generalised eigenvalues (for diagnostics)
%       stab:  1 = BK conditions satisfied (unique stable solution)
%              0 = indeterminate
%             -1 = explosive / no stable solution
%
%   Reference:
%       Klein, P. (2000). "Using the Generalized Schur Form to Solve a
%       Multivariate Linear Rational Expectations Model."
%       Journal of Economic Dynamics and Control, 24, 1405-1423.
%
%       CSD Toolbox (Duineveld, 2021), adapted from CoRRAM-M (Maussner, 2018).
%
%   See also: quantecon.dsge.Gensys, quantecon.dsge.KalmanFilter

arguments
    A   (:,:) double
    B   (:,:) double
    nx  (1,1) double {mustBeInteger, mustBeNonnegative}
    options.nz    (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Rho   (:,:) double = []
    options.Omega (:,:) double = []
end

n = size(A, 1);
nz = options.nz;
nw = nx + nz;      % total state dimension
ny = n - nw;        % number of controls

assert(size(A,2) == n && all(size(B) == [n, n]), ...
    'A and B must be square matrices of the same size.');
assert(nw + ny == n, 'nx + nz + ny must equal n = size(A,1).');

% --- QZ decomposition and reordering ---
[S, T, Q, Z] = qz(A, B);
[S, T, Q, Z] = ordqz(S, T, Q, Z, 'udo');  % stable roots upper-left

% --- Extract blocks ---
z21 = Z(nw+1:end, 1:nw);
z11 = Z(1:nw, 1:nw);

if rank(z11) < nw
    stab = -1;
    SOL.F = NaN(ny, nw);
    SOL.P = NaN(nw, nw);
    SOL.Hx_w = NaN(nx, nw);
    SOL.Hy_w = SOL.F;
    SOL.gev = [diag(S), diag(T)];
    warning('PertSolve:singularZ11', 'Invertibility condition violated. rank(z11) < nw.');
    return
end

s11 = S(1:nw, 1:nw);
t11 = T(1:nw, 1:nw);

% --- Blanchard-Kahn check ---
% Stable eigenvalues should be in upper-left block (nw of them)
% Check: |t(nw,nw)/s(nw,nw)| < 1 and |t(nw+1,nw+1)/s(nw+1,nw+1)| > 1
if nw < n
    last_stable = abs(T(nw, nw)) / max(abs(S(nw, nw)), eps);
    first_unstable = abs(T(nw+1, nw+1)) / max(abs(S(nw+1, nw+1)), eps);
    if last_stable > 1
        stab = -1;  % explosive
        SOL.F = NaN(ny, nw);
        SOL.P = NaN(nw, nw);
        SOL.Hx_w = NaN(nx, nw);
        SOL.Hy_w = SOL.F;
        SOL.gev = [diag(S), diag(T)];
        warning('PertSolve:explosive', 'Steady state is locally explosive.');
        return
    end
    if first_unstable < 1
        stab = 0;  % indeterminate
        SOL.F = NaN(ny, nw);
        SOL.P = NaN(nw, nw);
        SOL.Hx_w = NaN(nx, nw);
        SOL.Hy_w = SOL.F;
        SOL.gev = [diag(S), diag(T)];
        warning('PertSolve:indeterminate', 'Model is indeterminate.');
        return
    end
end

stab = 1;

% --- Build solution ---
% [FIX]: use z11\ instead of inv(z11)
z11i = z11 \ eye(nw);
dyn = s11 \ t11;

F = real(z21 * z11i);       % decision rule: u = F * w
P = real(z11 * dyn * z11i); % law of motion: w' = P * w

SOL.F     = F;
SOL.P     = P;
SOL.Hx_w  = P(1:nx, :);
SOL.Hy_w  = F;
SOL.gev   = [diag(S), diag(T)];

% --- Store Rho/Omega if provided ---
if ~isempty(options.Rho)
    SOL.Rho = options.Rho;
end
if ~isempty(options.Omega)
    SOL.Omega = options.Omega;
end

end
