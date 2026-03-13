function X = DiscLyap(A, C, options)
%DISCLYAP Solve the discrete Lyapunov equation via the doubling algorithm.
%
%   Solves:   X = A * X * A' + C
%
%   using the matrix doubling method. This is often more numerically
%   stable than MATLAB's built-in dlyap for large or near-singular systems,
%   and avoids the need for the Control System Toolbox.
%
%   INPUTS:
%       A - (n x n) transition matrix (spectral radius < 1 for convergence)
%       C - (n x n) driving matrix (typically positive semi-definite)
%
%   Name-Value Options:
%       Tol     - convergence tolerance.  Default: 1e-12
%       MaxIter - maximum iterations.     Default: 500
%       Sym     - enforce symmetry on C.  Default: true
%
%   OUTPUT:
%       X - (n x n) solution matrix
%
%   Reference:
%       Anderson, B.D.O. & Moore, J.B. (1979). "Optimal Filtering."
%       Doubling algorithm from Mutschler (2018), Econometrics & Statistics.
%
%   See also: quantecon.dsge.KalmanFilter, dlyap

arguments
    A       (:,:) double
    C       (:,:) double
    options.Tol     (1,1) double {mustBePositive} = 1e-12
    options.MaxIter (1,1) double {mustBePositive, mustBeInteger} = 500
    options.Sym     (1,1) logical = true
end

n = size(A, 1);
assert(size(A,2) == n && all(size(C) == [n, n]), ...
    'A and C must be n x n matrices of the same size.');

tol = options.Tol;
maxiter = options.MaxIter;

% Enforce symmetry if requested
if options.Sym
    C = 0.5 * (C + C');
end

% Doubling iteration: converges in O(log n) steps
X = C;
Ak = A;
for iter = 1:maxiter
    X_new = Ak * X * Ak' + X;
    diff = max(abs(X_new - X), [], 'all');
    if diff < tol
        X = X_new;
        break
    end
    % Update: A_{k+1} = A_k * A_k,  C_{k+1} = A_k * C_k * A_k' + C_k
    Ak = Ak * Ak;
    X = X_new;
end

if iter == maxiter && diff >= tol
    warning('DiscLyap:noConverge', ...
        'Doubling algorithm did not converge in %d iterations (diff=%.2e).', maxiter, diff);
end

% Enforce symmetry on output
if options.Sym
    X = 0.5 * (X + X');
end

end
