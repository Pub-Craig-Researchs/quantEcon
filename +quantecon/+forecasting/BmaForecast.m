function forecast = BmaForecast(forecast_cell, weights)
%BMAFORECAST Bayesian Model Averaging for Forecasts
%
%   forecast_cell: {N_models x 1} each [T_horizon x K]
%   weights: [N_models x 1] prior/posterior model probabilities

arguments
    forecast_cell cell
    weights (:,1) double
end

% Normalize weights
weights = weights / sum(weights);
nModels = length(forecast_cell);
[H, K] = size(forecast_cell{1});

forecast = zeros(H, K);
for i = 1:nModels
    forecast = forecast + weights(i) * forecast_cell{i};
end
end
