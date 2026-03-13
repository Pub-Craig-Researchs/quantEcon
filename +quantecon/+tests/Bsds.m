function [c, u, l] = Bsds(bench, models, opts)
%BSDS Bootstrap Data Snooping — White's Reality Check & Hansen's SPA test
%
%   Tests whether any competing model significantly outperforms a
%   benchmark, accounting for the multiplicity of comparisons.
%   Returns three p-values corresponding to different treatments of
%   poor models:
%     - Consistent (c): Hansen's SPA — shrinks contributions of models
%       far worse than benchmark (recommended).
%     - Upper (u): White's original Reality Check — no shrinkage.
%     - Lower (l): Aggressive shrinkage — sets all worse models to zero.
%
%   H0: The benchmark is as good as the best alternative.
%   H1: At least one model has smaller expected loss than the benchmark.
%
%   Reference:
%       White, H. (2000). "A Reality Check for Data Snooping."
%       Econometrica 68(5), 1097-1126.
%
%       Hansen, P.R. (2005). "A Test for Superior Predictive Ability."
%       Journal of Business & Economic Statistics 23(4), 365-380.
%
%   Usage:
%       [c, u, l] = quantecon.tests.Bsds(benchLoss, modelLosses);
%       [c, u, l] = quantecon.tests.Bsds(benchLoss, modelLosses, ...
%                       'B', 1000, 'BlockLen', 12, 'Type', 'studentized');
%
%   Inputs:
%       bench  - (T x 1) loss series from the benchmark model
%       models - (T x K) loss series from K competing models
%
%   Options:
%       'B'        - (int)    Number of bootstrap replications (default: 1000)
%       'BlockLen' - (int)    Block length for bootstrap (default: max(1,floor(T^(1/3))))
%       'Type'     - (string) 'studentized' (default) or 'standard'
%       'Boot'     - (string) 'stationary' (default) or 'block'
%
%   Outputs:
%       c - Hansen's consistent SPA p-value (recommended)
%       u - White's Reality Check (upper) p-value
%       l - Lower (most aggressive) SPA p-value
%
%   Note:
%       Inputs should be LOSSES (bads). For "goods" (e.g. returns),
%       pass -bench and -models.
%
%   See also quantecon.tests.Mcs

arguments
    bench  (:,1) double
    models (:,:) double
    opts.B        (1,1) double {mustBePositive, mustBeInteger} = 1000
    opts.BlockLen (1,1) double {mustBeNonnegative, mustBeInteger} = 0  % 0 = auto
    opts.Type     (1,1) string {mustBeMember(opts.Type, ["studentized","standard"])} = "studentized"
    opts.Boot     (1,1) string {mustBeMember(opts.Boot, ["stationary","block"])} = "stationary"
end

[T, K] = size(models);

if length(bench) ~= T
    error('quantecon:tests:Bsds:sizeMismatch', ...
          'BENCH and MODELS must have the same number of rows.');
end

B = opts.B;
w = opts.BlockLen;
if w == 0
    w = max(1, floor(T^(1/3)));
end

isStudentized = (opts.Type == "studentized");

% ---- Generate bootstrap index matrix (T x B) ----
if opts.Boot == "block"
    [~, bsIdx] = quantecon.base.BlockBootstrap((1:T)', B, w);
else
    [~, bsIdx] = quantecon.base.StationaryBootstrap((1:T)', B, w);
end

% ---- Loss differentials: d_ik = model_k - bench ----
diffs = models - bench;  % (T x K)

% ---- HAC-style variance estimate (flat-top kernel) ----
q = 1 / w;
lag = (1:T-1)';
kappa = ((T - lag) ./ T) .* (1 - q).^lag + (lag ./ T) .* (1 - q).^(T - lag);

vars = zeros(1, K);
for k = 1:K
    dk = diffs(:, k) - mean(diffs(:, k));
    vars(k) = (dk' * dk) / T;
    for j = 1:T-1
        vars(k) = vars(k) + 2 * kappa(j) * (dk(1:T-j)' * dk(j+1:T)) / T;
    end
end
% Guard against non-positive variance
vars = max(vars, eps);

% ---- Thresholds for re-centering ----
dbar = mean(diffs, 1);                          % (1 x K)
Anew = sqrt((vars / T) * 2 * log(log(max(T, 3))));

% Consistent: re-center only if dbar is small (or model is better)
gc = dbar .* (dbar < Anew);
% Lower: re-center only if model is worse (dbar > 0 means model worse)
gl = min(0, dbar);
% Upper: no re-centering
gu = dbar;

% ---- Studentization factor ----
if isStudentized
    stdDev = sqrt(vars);
else
    stdDev = ones(1, K);
end

% ---- Bootstrap distribution of min statistic ----
perfc = zeros(B, 1);
perfl = zeros(B, 1);
perfu = zeros(B, 1);

for b = 1:B
    % Bootstrap mean of each loss differential
    mboot = mean(diffs(bsIdx(:, b), :), 1);  % (1 x K)

    % Re-centered and studentized bootstrap statistics
    tc = (mboot - gc) ./ stdDev;
    tl = (mboot - gl) ./ stdDev;
    tu = (mboot - gu) ./ stdDev;

    perfc(b) = min(tc);
    perfl(b) = min(tl);
    perfu(b) = min(tu);
end

% Truncate at zero (only negative values count as "better than benchmark")
perfc = min(perfc, 0);
perfl = min(perfl, 0);
perfu = min(perfu, 0);

% ---- Test statistic ----
stat = min(dbar ./ stdDev);

% ---- P-values ----
c = mean(perfc < stat);
l = mean(perfl < stat);
u = mean(perfu < stat);

end
