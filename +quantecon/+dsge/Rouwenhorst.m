function [grid, P] = Rouwenhorst(rho, sigma, options)
%ROUWENHORST Rouwenhorst (1995) AR(1) discretisation.
%
%   Approximates the AR(1) process:
%       y(t) = rho * y(t-1) + sigma * e(t),  e(t) ~ N(0,1)
%
%   by a finite-state Markov chain with N states. More accurate than
%   Tauchen for highly persistent processes (rho close to 1).
%
%   INPUTS:
%       rho   - (scalar) persistence parameter, |rho| < 1
%       sigma - (scalar) standard deviation of innovation
%       N     - (optional, default 7) number of grid points (>= 2)
%
%   OUTPUTS:
%       grid  - (N x 1) state space grid
%       P     - (N x N) transition probability matrix (rows sum to 1)
%
%   Reference:
%       Rouwenhorst, K.G. (1995). "Asset Pricing Implications of
%       Equilibrium Business Cycle Models." In T. Cooley (Ed.), Frontiers
%       of Business Cycle Research, Princeton University Press.
%
%   See also: quantecon.dsge.Tauchen

arguments
    rho   (1,1) double {mustBeInRange(rho, -1, 1, 'exclusive')}
    sigma (1,1) double {mustBePositive}
    options.N (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.N, 2)} = 7
end

N = options.N;
p = (1 + rho) / 2;
q = p;

% Unconditional standard deviation
sigma_y = sigma / sqrt(1 - rho^2);

% Grid: equally spaced on [-psi, psi] where psi = sqrt(N-1)*sigma_y
psi = sqrt(N - 1) * sigma_y;
grid = linspace(-psi, psi, N)';

% Build transition matrix recursively
% Base case: N = 2
P_prev = [p, 1-p; 1-q, q];

for n = 3:N
    z = zeros(n-1, 1);
    P_new = p     * [P_prev, z; z', 0] + ...
            (1-p) * [z, P_prev; 0, z'] + ...
            (1-q) * [z', 0; P_prev, z] + ...
            q     * [0, z'; z, P_prev];
    % Normalise interior rows (divide by 2)
    P_new(2:end-1, :) = P_new(2:end-1, :) / 2;
    P_prev = P_new;
end

P = P_prev;

end
