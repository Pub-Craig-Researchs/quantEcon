function results = DieboldMariano(e1, e2, opts)
%DIEBOLDMARIANO Diebold-Mariano Test for Predictive Accuracy
%
%   Usage:
%       results = quantecon.base.DieboldMariano(e1, e2)
%       results = quantecon.base.DieboldMariano(e1, e2, 'LossType', 'MSE', 'Lag', 0)
%
%   Inputs:
%       e1 - (T x 1) Forecast errors from model 1
%       e2 - (T x 1) Forecast errors from model 2
%
%   Options:
%       'LossType' - "MSE" (Default) or "MAE"
%       'Lag'      - (int) Lag for Newey-West variance estimator (Default: automatic)
%
%   Outputs:
%       results - Struct with DM statistic and p-value.

arguments
    e1 (:,1) double {mustBeNumeric, mustBeReal}
    e2 (:,1) double {mustBeNumeric, mustBeReal}
    opts.LossType (1,1) string {mustBeMember(opts.LossType, ["MSE", "MAE"])} = "MSE"
    opts.Lag = []
end

T = length(e1);
if length(e2) ~= T
    error("quantecon:base:DieboldMariano:DimensionMismatch", "Input error vectors must have the same length.");
end

% Define loss function
if strcmpi(opts.LossType, "MSE")
    d = e1.^2 - e2.^2;
elseif strcmpi(opts.LossType, "MAE")
    d = abs(e1) - abs(e2);
end

d_bar = mean(d);

L = opts.Lag;
if isempty(L)
    L = floor(4 * (T/100)^(2/9));
end

% Autocovariances up to lag L
gamma = zeros(L + 1, 1);
for l = 0:L
    c = 0;
    for t = l+1:T
        c = c + (d(t) - d_bar) * (d(t-l) - d_bar);
    end
    gamma(l+1) = c / T;
end

% Long-run variance (HAC)
V = gamma(1);
for j = 1:L
    w_j = 1 - j / (L + 1);
    V = V + 2 * w_j * gamma(j+1);
end

dm_stat = d_bar / sqrt(V / T);
p_value = 2 * (1 - normcdf(abs(dm_stat)));

results.Statistic = dm_stat;
results.pValue = p_value;
results.dBar = d_bar;
results.Variance = V;
results.Lag = L;
results.LossType = opts.LossType;

end
