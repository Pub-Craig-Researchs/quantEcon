function [x, w] = GaussHermite(n)
%GAUSSHERMITE Gauss-Hermite quadrature nodes and weights
%
%   Computes n-point Gauss-Hermite quadrature for integrals of the form
%       integral exp(-x^2) f(x) dx  ~=  sum w_i * f(x_i)
%
%   Uses the three-term recurrence of the (probabilists') Hermite
%   polynomials via symmetric tridiagonal eigenvalue decomposition
%   (Golub-Welsch algorithm).
%
%   Reference:
%       Golub, G.H. & Welsch, J.H. (1969). "Calculation of Gauss
%       Quadrature Rules." Math. Comp. 23, 221-230.
%
%   Usage:
%       [x, w] = quantecon.bayes.GaussHermite(5);
%
%   Inputs:
%       n - (int) number of quadrature points (n >= 1)
%
%   Outputs:
%       x - (n x 1) nodes (roots of H_n)
%       w - (n x 1) weights (sum = sqrt(pi))

arguments
    n (1,1) double {mustBePositive, mustBeInteger}
end

if n == 1
    x = 0;
    w = sqrt(pi);
    return
end

% Golub-Welsch: symmetric tridiagonal matrix whose eigenvalues are
% the nodes of Gauss-Hermite quadrature (physicists' convention).
% Sub-diagonal entries: beta_i = sqrt(i/2), i = 1,...,n-1
beta = sqrt((1:n-1) / 2)';
J    = diag(beta, -1) + diag(beta, 1);

[V, D] = eig(J, 'vector');
[x, idx] = sort(D);
V = V(:, idx);

% Weights from first component of eigenvectors
w = sqrt(pi) * V(1, :)'.^2;

x = x(:);
w = w(:);
end
