function result = MhSampler(logTarget, theta0, opts)
% MhSampler  Generic Metropolis-Hastings MCMC sampler
%
%   Random-walk MH with multivariate normal proposal, or user-supplied
%   proposal. Supports burn-in, thinning, and adaptive proposal scaling.
%
% SYNTAX:
%   result = quantecon.bayes.MhSampler(@logpdf, theta0)
%   result = quantecon.bayes.MhSampler(@logpdf, theta0, NSamples=5000)
%
% INPUT:
%   logTarget - function handle @(theta) returning log-posterior (up to constant)
%   theta0    - (d x 1) initial parameter vector
%
% OPTIONS:
%   NSamples  - number of retained samples  (default 5000)
%   BurnIn    - burn-in draws to discard  (default 1000)
%   Thin      - thinning interval  (default 1)
%   PropCov   - (d x d) proposal covariance  (default eye(d))
%   PropScale - scalar multiplier for proposal  (default 2.38^2/d)
%   Adapt     - logical, enable adaptive scaling during burn-in  (default true)
%
% OUTPUT:
%   result struct with fields:
%     .chain     - (NSamples x d) posterior samples
%     .logpost   - (NSamples x 1) log-posterior values
%     .accept    - scalar acceptance rate
%     .PropCov   - final proposal covariance used
%
% REFERENCE:
%   Metropolis, N. et al. (1953) "Equation of state calculations by
%   fast computing machines", J. Chemical Physics, 21, 1087-1092.
%   Roberts, G.O. & Rosenthal, J.S. (2001) "Optimal scaling for
%   various Metropolis-Hastings algorithms", Statistical Science, 16, 351-367.
%
% Source: Generic MH implementation following Bayesian computation standards.

arguments
    logTarget function_handle
    theta0 (:,1) double
    opts.NSamples (1,1) double {mustBePositive, mustBeInteger} = 5000
    opts.BurnIn (1,1) double {mustBeNonnegative, mustBeInteger} = 1000
    opts.Thin (1,1) double {mustBePositive, mustBeInteger} = 1
    opts.PropCov double = []
    opts.PropScale (1,1) double {mustBeNonnegative} = 0
    opts.Adapt (1,1) logical = true
end

d = length(theta0);
nTotal = opts.BurnIn + opts.NSamples * opts.Thin;

% Default proposal covariance
if isempty(opts.PropCov)
    Sigma = eye(d);
else
    Sigma = opts.PropCov;
end
if opts.PropScale == 0
    sc = 2.38^2 / d;   % Roberts & Rosenthal (2001) optimal scaling
else
    sc = opts.PropScale;
end

% Cholesky for sampling
L = chol(sc * Sigma, 'lower');

% Pre-allocate
chain   = zeros(opts.NSamples, d);
logpost = zeros(opts.NSamples, 1);
nAccept = 0;
nStore  = 0;

theta = theta0;
lp    = logTarget(theta);

for i = 1:nTotal
    % Propose
    theta_star = theta + L * randn(d, 1);
    lp_star    = logTarget(theta_star);

    % Accept/reject (log scale for numerical stability)
    log_alpha = lp_star - lp;
    if log(rand) < min(log_alpha, 0)
        theta = theta_star;
        lp    = lp_star;
        nAccept = nAccept + 1;
    end

    % Adaptive scaling during burn-in
    if opts.Adapt && i <= opts.BurnIn && mod(i, 100) == 0
        rate = nAccept / i;
        if rate < 0.15
            sc = sc * 0.5;
        elseif rate > 0.50
            sc = sc * 1.5;
        end
        L = chol(sc * Sigma, 'lower');
    end

    % Store after burn-in with thinning
    if i > opts.BurnIn && mod(i - opts.BurnIn, opts.Thin) == 0
        nStore = nStore + 1;
        chain(nStore, :)  = theta';
        logpost(nStore)   = lp;
    end
end

result = struct( ...
    'chain',   chain, ...
    'logpost', logpost, ...
    'accept',  nAccept / nTotal, ...
    'PropCov', sc * Sigma);
end
