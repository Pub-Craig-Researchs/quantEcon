function results = AwmlFit(x, opts)
%AWMLFIT Approximate Whittle MLE for fractional OU process
%
%   Usage:
%       res = quantecon.timeseries.fou.AwmlFit(x)
%       res = quantecon.timeseries.fou.AwmlFit(x, "K", 60, "Delta", 1/252)
%
%   Inputs:
%       x - (T x 1) series
%
%   Options:
%       K            - Truncation parameter for spectral density (default: 50)
%       Delta        - Sampling interval (default: 1)
%       Initial      - Initial guess [H, kappa]
%       HBounds      - [Hmin, Hmax]
%       KappaBounds  - [kmin, kmax]
%       Seed         - RNG seed (default: 0)
%       Infer        - Whether to compute standard errors (default: true)
%
%   Output:
%       results - struct with fields: Params, Sigma2, StdErrors, LogLike
%
%   Reference:
%       Shi, S., Yu, J., Zhang, C. (2024), Journal of Econometrics.

arguments
    x (:,1) double
    opts.K (1,1) double {mustBeInteger, mustBePositive} = 50
    opts.Delta (1,1) double {mustBePositive} = 1
    opts.Initial (1,2) double = [0.4, 1]
    opts.HBounds (1,2) double = [1.0e-3, 0.999]
    opts.KappaBounds (1,2) double = [1.0e-6, 100]
    opts.Seed (1,1) double = 0
    opts.Infer (1,1) logical = true
end

rng(opts.Seed, "twister");

x = x(:);
if any(~isfinite(x))
    error("quantecon:fou:AwmlFit:InvalidData", "x contains NaN/Inf.");
end

lb = [opts.HBounds(1), opts.KappaBounds(1)];
ub = [opts.HBounds(2), opts.KappaBounds(2)];

objective = @(p) quantecon.timeseries.fou.AwmlLikelihood(x, p, opts.K, opts.Delta);

options = optimoptions("fmincon", "Display", "off", "Algorithm", "sqp");
params = fmincon(objective, opts.Initial, [], [], [], [], lb, ub, [], options);

sigma2 = quantecon.timeseries.fou.AwmlSigma2(x, params, opts.K, opts.Delta);
loglike = -objective(params);

results = struct();
results.Params = params;
results.Sigma2 = sigma2;
results.LogLike = loglike;

if opts.Infer
    se = quantecon.timeseries.fou.Infer(params(1), params(2), sqrt(sigma2), length(x), opts.K, opts.Delta);
    results.StdErrors = se(:)';
else
    results.StdErrors = [];
end
end
