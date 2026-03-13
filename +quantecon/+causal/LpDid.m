function results = LpDid(Y, D, unitID, timeID, opts)
%LPDID Local Projections Difference-in-Differences (LP-DiD)
%
%   Estimates dynamic treatment effects via LP-DiD, avoiding the
%   negative-weighting bias of two-way FE estimators. Restricts
%   estimation to newly treated units and clean controls.
%
%   Usage:
%       results = LpDid(Y, D, unitID, timeID)
%       results = LpDid(Y, D, unitID, timeID, 'PostWindow', 5, ...
%                   'PreWindow', 3, 'Weighting', 'vw')
%
%   Inputs:
%       Y       - (NT x 1) double, outcome variable (stacked panel)
%       D       - (NT x 1) double, binary treatment indicator
%       unitID  - (NT x 1) int/double, unit identifier
%       timeID  - (NT x 1) int/double, time period identifier
%
%   Options (Name-Value):
%       PostWindow  - int, horizons after treatment (default: 5)
%       PreWindow   - int, pre-treatment horizons for testing (default: 3)
%       Weighting   - string, 'vw' (variance-weighted, default),
%                     'rw' (reweighted/equally-weighted ATT), or 'both'
%       Controls    - (NT x k) double, control variables (default: [])
%       Alpha       - double, significance level (default: 0.05)
%
%   Output:
%       results - struct with fields:
%           .Horizons - (H x 1) horizon vector [-PreWindow..PostWindow]
%           .BetaVW   - (H x 1) VW LP-DiD coefficients
%           .SeVW     - (H x 1) cluster-robust SEs (VW)
%           .CiLoVW   - (H x 1) lower CI (VW)
%           .CiHiVW   - (H x 1) upper CI (VW)
%           .BetaRW   - (H x 1) RW LP-DiD coefficients (if Weighting='rw'/'both')
%           .SeRW     - (H x 1) cluster-robust SEs (RW)
%           .CiLoRW   - (H x 1) lower CI (RW)
%           .CiHiRW   - (H x 1) upper CI (RW)
%
%   Reference:
%       Dube, Girardi, Jorda & Taylor (2025), "A Local Projections
%       Approach to Difference-in-Differences", JAE 40(7).
%
%   See also: quantecon.timeseries.LocalProjection,
%             quantecon.causal.Did

arguments
    Y (:,1) double
    D (:,1) double
    unitID (:,1) double
    timeID (:,1) double
    opts.PostWindow (1,1) double {mustBeNonnegative, mustBeInteger} = 5
    opts.PreWindow (1,1) double {mustBeNonnegative, mustBeInteger} = 3
    opts.Weighting (1,1) string {mustBeMember(opts.Weighting, ...
        ["vw","rw","both"])} = "both"
    opts.Controls double = []
    opts.Alpha (1,1) double = 0.05
end

H_post = opts.PostWindow;
H_pre  = opts.PreWindow;
alpha  = opts.Alpha;
doVW   = opts.Weighting ~= "rw";
doRW   = opts.Weighting ~= "vw";
W      = opts.Controls; %#ok<NASGU> reserved for future covariate adjustment

cv = norminv(1 - alpha/2);

% Build balanced panel lookup: unit -> sorted time -> Y, D
units = unique(unitID);
times = unique(timeID);
nU = numel(units);
nT = numel(times);

% Map to integer indices
[~, uIdx] = ismember(unitID, units);
[~, tIdx] = ismember(timeID, times);

% Store in (nU x nT) arrays
Ymat = NaN(nU, nT);
Dmat = NaN(nU, nT);
for i = 1:length(Y)
    Ymat(uIdx(i), tIdx(i)) = Y(i);
    Dmat(uIdx(i), tIdx(i)) = D(i);
end

% Compute Delta D: D_{it} - D_{i,t-1}
deltaD = [NaN(nU, 1), diff(Dmat, 1, 2)];

% Horizon vector
horzs = [-H_pre:-2, 0:H_post];  % skip h=-1 (reference)
nH = numel(horzs);

% Results
betaVW = NaN(nH, 1); seVW = NaN(nH, 1);
betaRW = NaN(nH, 1); seRW = NaN(nH, 1);

for hh = 1:nH
    h = horzs(hh);

    % Build dependent variable and sample restriction
    if h >= 0
        % Post: y_{i,t+h} - y_{i,t-1}
        % Clean control: D_{i,t+h} = 0
        dep = NaN(nU, nT);
        keep = false(nU, nT);
        for t = 2:nT
            if t + h <= nT
                dep(:, t) = Ymat(:, t + h) - Ymat(:, t - 1);
            end
            for u = 1:nU
                if t + h <= nT && ~isnan(deltaD(u, t))
                    isTreated = (deltaD(u, t) == 1);
                    isClean   = (Dmat(u, t + h) == 0);
                    keep(u, t) = isTreated || isClean;
                end
            end
        end
    else
        % Pre: y_{i,t+h} - y_{i,t-1} = y_{i,t-|h|} - y_{i,t-1}
        ah = abs(h);
        dep = NaN(nU, nT);
        keep = false(nU, nT);
        for t = 2:nT
            if t - ah >= 1
                dep(:, t) = Ymat(:, t - ah) - Ymat(:, t - 1);
            end
            for u = 1:nU
                if t - ah >= 1 && ~isnan(deltaD(u, t))
                    isTreated = (deltaD(u, t) == 1);
                    isClean   = (Dmat(u, t) == 0);  % never treated at t for pre
                    keep(u, t) = isTreated || isClean;
                end
            end
        end
    end

    % Flatten to vectors
    [uAll, tAll] = ndgrid(1:nU, 1:nT);
    uVec = uAll(:); tVec = tAll(:);
    depVec = dep(:);
    ddVec  = deltaD(:);
    keepVec = keep(:);

    valid = keepVec & ~isnan(depVec) & ~isnan(ddVec);
    depV = depVec(valid);
    ddV  = ddVec(valid);
    uV   = uVec(valid);
    tV   = tVec(valid);

    if sum(valid) < 3; continue; end

    % VW: OLS with time FE, clustered by unit
    if doVW
        [b, se] = lpdid_fe_cluster(depV, ddV, tV, uV, []);
        betaVW(hh) = b;
        seVW(hh)   = se;
    end

    % RW: inverse propensity weights for equally-weighted ATT
    if doRW
        % Compute weights: residualize deltaD on time FE among treated+controls
        w = lpdid_rw_weights(ddV, tV, uV);
        [b, se] = lpdid_fe_cluster(depV, ddV, tV, uV, w);
        betaRW(hh) = b;
        seRW(hh)   = se;
    end
end

% Add h=-1 reference point
allH = [-H_pre:-1, 0:H_post];
nAll = numel(allH);
refIdx = H_pre;  % position of h=-1

betaVW_full = NaN(nAll, 1); seVW_full = NaN(nAll, 1);
betaRW_full = NaN(nAll, 1); seRW_full = NaN(nAll, 1);

% Map horzs to allH indices
for hh = 1:nH
    pos = find(allH == horzs(hh));
    if doVW; betaVW_full(pos) = betaVW(hh); seVW_full(pos) = seVW(hh); end
    if doRW; betaRW_full(pos) = betaRW(hh); seRW_full(pos) = seRW(hh); end
end
betaVW_full(refIdx) = 0; betaRW_full(refIdx) = 0;

results.Horizons = allH(:);
results.BetaVW   = betaVW_full;
results.SeVW     = seVW_full;
results.CiLoVW   = betaVW_full - cv * seVW_full;
results.CiHiVW   = betaVW_full + cv * seVW_full;
results.BetaRW   = betaRW_full;
results.SeRW     = seRW_full;
results.CiLoRW   = betaRW_full - cv * seRW_full;
results.CiHiRW   = betaRW_full + cv * seRW_full;

end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [b, se] = lpdid_fe_cluster(y, x, tV, uV, w)
%LPDID_FE_CLUSTER OLS with time FE and clustered SEs

n = length(y);
tVals = unique(tV);
nT = numel(tVals);

% Build time dummies
timeDum = zeros(n, nT);
for j = 1:nT
    timeDum(:, j) = double(tV == tVals(j));
end

% Drop one time dummy to avoid perfect collinearity
X = [x, timeDum(:, 2:end)];

if ~isempty(w)
    sqrtW = sqrt(w);
    Xw = X .* sqrtW;
    yw = y .* sqrtW;
else
    Xw = X;
    yw = y;
end

bAll = (Xw' * Xw) \ (Xw' * yw);
b = bAll(1);

% Residuals (unweighted for cluster SEs)
resid = y - X * bAll;
if ~isempty(w); resid = resid .* sqrt(w); end

% Cluster-robust SEs by unit
uVals = unique(uV);
nC = numel(uVals);
k = size(X, 2);

XXinv = (Xw' * Xw) \ eye(k);
meat = zeros(k, k);
for c = 1:nC
    idx = (uV == uVals(c));
    if ~isempty(w)
        score_c = Xw(idx, :)' * resid(idx);
    else
        score_c = X(idx, :)' * resid(idx);
    end
    meat = meat + score_c * score_c';
end

% Small-sample correction
adj = nC / (nC - 1) * (n - 1) / (n - k);
V = XXinv * meat * XXinv * adj;
se = sqrt(max(V(1, 1), 0));
end

function w = lpdid_rw_weights(ddV, tV, ~)
%LPDID_RW_WEIGHTS Compute inverse propensity weights for reweighted LP-DiD
%   Follows Dube et al. (2025): residualize deltaD on time FE to get
%   propensity scores, then construct IPW weights.

n = length(ddV);
tVals = unique(tV);
nT = numel(tVals);

% Residualize deltaD on time FE
timeDum = zeros(n, nT);
for j = 1:nT
    timeDum(:, j) = double(tV == tVals(j));
end
Xt = timeDum(:, 2:end);
bT = (Xt' * Xt) \ (Xt' * ddV);
ddResid = ddV - Xt * bT;  % propensity score residuals

% Time-period propensity: P(deltaD=1 | t)
p_t = zeros(n, 1);
for j = 1:nT
    tMask = (tV == tVals(j));
    p_t(tMask) = mean(ddV(tMask));
end

% IPW weights per observation
isTreated = (ddV == 1);
w = zeros(n, 1);

% Treated: w_i = ddResid_i / sum(ddResid for treated)
ddR_treat_sum = sum(ddResid(isTreated));
if abs(ddR_treat_sum) < 1e-15
    % Fallback: equal weights
    w = ones(n, 1);
    return;
end
w(isTreated) = ddResid(isTreated) / ddR_treat_sum;

% Controls: w_i = -p_t/(1-p_t) * ddResid_i / sum(ddResid for treated)
% This re-weights controls to match treated distribution across time
isControl = ~isTreated;
p_c = p_t(isControl);
safe_p = max(p_c, 1e-10);
w(isControl) = (safe_p ./ (1 - safe_p)) .* (-ddResid(isControl)) / ddR_treat_sum;

% Ensure non-negative and normalize
w = abs(w);
w = w / sum(w) * n;
end
