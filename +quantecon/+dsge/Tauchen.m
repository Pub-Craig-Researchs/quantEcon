function [grid, P] = Tauchen(rho, sigma, options)
%TAUCHEN Tauchen (1986) / Tauchen-Hussey (1991) AR(1) discretisation.
%
%   Approximates the AR(1) process:
%       y(t) = rho * y(t-1) + sigma * e(t),  e(t) ~ N(0,1)
%
%   by a finite-state Markov chain with N states.
%
%   INPUTS:
%       rho   - (scalar) persistence parameter, |rho| < 1
%       sigma - (scalar) standard deviation of innovation
%       N     - (optional, default 7) number of grid points
%       M     - (optional, default 3) grid covers +/- M unconditional std devs
%
%   OUTPUTS:
%       grid  - (N x 1) state space grid
%       P     - (N x N) transition probability matrix (rows sum to 1)
%
%   Reference:
%       Tauchen, G. (1986). "Finite State Markov-Chain Approximations to
%       Univariate and Vector Autoregressions." Economics Letters, 20, 177-181.
%
%   See also: quantecon.dsge.Rouwenhorst

arguments
    rho   (1,1) double {mustBeInRange(rho, -1, 1, 'exclusive')}
    sigma (1,1) double {mustBePositive}
    options.N (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.N, 2)} = 7
    options.M (1,1) double {mustBePositive} = 3
end

N = options.N;
M = options.M;

% Unconditional standard deviation
sigma_y = sigma / sqrt(1 - rho^2);

% Grid: equally spaced on [-M*sigma_y, M*sigma_y]
grid = linspace(-M * sigma_y, M * sigma_y, N)';
d = grid(2) - grid(1);  % step size

% Build transition matrix using normal CDF
P = zeros(N, N);
for i = 1:N
    % Conditional mean: rho * grid(i)
    mu = rho * grid(i);
    % Interior columns
    for j = 2:N-1
        P(i, j) = normcdf((grid(j) + d/2 - mu) / sigma) ...
                 - normcdf((grid(j) - d/2 - mu) / sigma);
    end
    % Boundary columns absorb the tails
    P(i, 1) = normcdf((grid(1) + d/2 - mu) / sigma);
    P(i, N) = 1 - normcdf((grid(N) - d/2 - mu) / sigma);
end

end
