function results = MurphyDiagram(y, f1, f2, opts)
%MURPHYDIAGRAM Visual comparison of forecasting models
%
%   Computes Elementary Scoring Functions to compare two point forecasts.
%   Reference: Ehm et al. (2016).
%
%   Usage:
%       res = quantecon.forecasting.MurphyDiagram(y, f1, f2);

arguments
    y (:,1) double % Actual
    f1 (:,1) double % Forecast 1
    f2 (:,1) double % Forecast 2
    opts.Theta (:,1) double = []
    opts.Quantile (1,1) double = 0.5
end

T = length(y);

% Grid for Theta (the threshold)
if isempty(opts.Theta)
    all_vals = [y; f1; f2];
    theta_grid = linspace(min(all_vals), max(all_vals), 100)';
else
    theta_grid = opts.Theta;
end

n_theta = length(theta_grid);
score1 = zeros(n_theta, 1);
score2 = zeros(n_theta, 1);

alpha = opts.Quantile;

% Elementary Scoring Function for Quantiles
% S(x, y, theta) = (1(y <= theta) - alpha) * (1(x <= theta) - 1(y <= theta))
% Note: Simplified version for Mean/Quantile
for i = 1:n_theta
    theta = theta_grid(i);

    % S_theta(f, y)
    val1 = (y <= theta) - alpha;
    val2 = (f1 <= theta) - alpha;
    val3 = (f2 <= theta) - alpha;

    % The score is usually integrated/averaged over time
    % Ehm et al. (2016) Eq 3.
    score1(i) = mean(abs((y <= theta) - alpha) .* abs(f1 - y) .* ( (f1 > theta & y <= theta) | (f1 <= theta & y > theta) ));
    % More commonly:
    score1(i) = mean( elementary_score(f1, y, theta, alpha) );
    score2(i) = mean( elementary_score(f2, y, theta, alpha) );
end

results.Theta = theta_grid;
results.Score1 = score1;
results.Score2 = score2;
results.Difference = score1 - score2;

% Find regions of dominance
results.Better1 = find(score1 < score2);
results.Better2 = find(score2 < score1);
end

function s = elementary_score(x, y, theta, alpha)
% S(x, y, theta) = |1(y <= theta) - alpha| * 1(min(x, y) <= theta < max(x, y))
ind = (min(x, y) <= theta) & (theta < max(x, y));
s = abs((y <= theta) - alpha) .* ind;
end
