function result = HestonPrice(opts)
% HestonPrice  European call option pricing under Heston stochastic volatility
%
%   Monte Carlo with conditional Black-Scholes (time-averaged variance)
%   and antithetic variates for variance reduction.
%
% SYNTAX:
%   result = quantecon.finance.HestonPrice(S0=100, Strike=90:5:110)
%
% OPTIONS (all name-value):
%   S0     - current spot price  (default 100)
%   r      - risk-free rate (annualised)  (default 0.02)
%   V0     - current variance  (default 0.04)
%   Eta    - volatility of volatility  (default 0.7)
%   Theta  - long-run variance  (default 0.06)
%   Kappa  - mean-reversion speed  (default 1.5)
%   Strike - vector of strike prices  (default 100)
%   T      - time to maturity in years  (default 0.25)
%   NPaths - number of MC paths  (default 10000)
%   NSteps - number of time steps per path  (default 250)
%
% OUTPUT:
%   result struct with fields:
%     .Price   - (nK x 1) call prices per strike
%     .StdErr  - (nK x 1) Monte Carlo standard errors
%     .Strike  - strike vector used
%
% REFERENCE:
%   Heston, S.L. (1993) "A closed-form solution for options with
%   stochastic volatility with applications to bond and currency
%   options", Review of Financial Studies, 6(2), 327-343.
%
% Source: Refactored from Sitter (U. Chicago) Heston.m.

arguments
    opts.S0 (1,1) double {mustBePositive} = 100
    opts.r (1,1) double = 0.02
    opts.V0 (1,1) double {mustBePositive} = 0.04
    opts.Eta (1,1) double {mustBePositive} = 0.7
    opts.Theta (1,1) double {mustBePositive} = 0.06
    opts.Kappa (1,1) double {mustBePositive} = 1.5
    opts.Strike (1,:) double {mustBePositive} = 100
    opts.T (1,1) double {mustBePositive} = 0.25
    opts.NPaths (1,1) double {mustBePositive, mustBeInteger} = 10000
    opts.NSteps (1,1) double {mustBePositive, mustBeInteger} = 250
end

S0    = opts.S0;
r     = opts.r;
V0    = opts.V0;
eta   = opts.Eta;
theta = opts.Theta;
kappa = opts.Kappa;
K     = opts.Strike(:);
tau   = opts.T;
M     = opts.NPaths;
N     = opts.NSteps;
dt    = tau / N;
sqdt  = sqrt(dt);

% --- Variance path simulation (Euler, absorption at 0) ---
V    = zeros(M, N+1);
Vneg = zeros(M, N+1);
V(:,1)    = V0;
Vneg(:,1) = V0;
W = randn(M, N);

for i = 1:N
    sV = sqrt(V(:,i));
    V(:,i+1) = V(:,i) + kappa*(theta - V(:,i))*dt + eta*sV.*W(:,i)*sqdt;
    V(:,i+1) = max(V(:,i+1), 0);  % [FIX]: absorption instead of multiplication

    sVn = sqrt(Vneg(:,i));
    Vneg(:,i+1) = Vneg(:,i) + kappa*(theta - Vneg(:,i))*dt - eta*sVn.*W(:,i)*sqdt;
    Vneg(:,i+1) = max(Vneg(:,i+1), 0);
end

% --- Time-averaged implied volatility (trapezoidal rule) ---
ImpVol    = sqrt((0.5*V(:,1) + 0.5*V(:,end) + sum(V(:,2:end-1),2)) * dt / tau);
ImpVolNeg = sqrt((0.5*Vneg(:,1) + 0.5*Vneg(:,end) + sum(Vneg(:,2:end-1),2)) * dt / tau);

% --- Black-Scholes pricing per strike ---
nK  = numel(K);
Price  = zeros(nK, 1);
StdErr = zeros(nK, 1);

for j = 1:nK
    sample = 0.5 * (bs_call(S0, K(j), tau, r, ImpVol) + ...
                     bs_call(S0, K(j), tau, r, ImpVolNeg));
    Price(j)  = mean(sample);
    StdErr(j) = std(sample) / sqrt(M);
end

result = struct('Price', Price, 'StdErr', StdErr, 'Strike', K);
end

% =====================================================================

function C = bs_call(S0, K, T, r, sigma)
% Vectorised Black-Scholes call price (sigma is M x 1 vector)
    F  = S0 * exp(r * T);
    d1 = log(F / K) ./ (sigma * sqrt(T)) + sigma * sqrt(T) / 2;
    d2 = d1 - sigma * sqrt(T);
    C  = exp(-r * T) * (F * normcdf(d1) - K * normcdf(d2));
end
