function results = PanelVar(Y_panel, lags, opts)
%PANELVAR Bayesian Panel VAR (Multi-unit)
%
%   Data: Y_panel is a cell array {N_units x 1}, each T x K
%
%   Implementation: Grouped OLS or Bayesian Pooled/Hierarchical.
%   Following BEAR sub-module logic for Panel VAR.

arguments
    Y_panel cell
    lags (1,1) double = 1
    opts.Type (1,1) string {mustBeMember(opts.Type, ["pooled", "fixed_effects"])} = "pooled"
end

nUnits = length(Y_panel);
[T, K] = size(Y_panel{1});

if opts.Type == "pooled"
    % Combine all data into one long VAR
    Y_stacked = [];
    X_stacked = [];
    for i = 1:nUnits
        Yi = Y_panel{i};
        [Ti, Ki] = size(Yi);
        Y_eff = Yi(lags+1:end, :);
        Xi = ones(Ti-lags, 1);
        for l = 1:lags
            Xi = [Xi, Yi(lags+1-l : end-l, :)];
        end
        Y_stacked = [Y_stacked; Y_eff];
        X_stacked = [X_stacked; Xi];
    end

    % Estimate pooled coefficients
    beta_pooled = (X_stacked' * X_stacked) \ (X_stacked' * Y_stacked);
    res = Y_stacked - X_stacked * beta_pooled;
    Sigma = (res' * res) / (size(Y_stacked, 1) - size(X_stacked, 2));

    results.Coefficients = beta_pooled;
    results.Sigma = Sigma;
    results.Type = "pooled";

elseif opts.Type == "fixed_effects"
    % Unit-specific intercepts, shared dynamics
    % Y_it = c_i + A1*Y_i,t-1 + ...

    % Number of params: K (intercepts) * nUnits + K*K*lags (dynamics)
    % This is a high-dimensional system.

    % Implementation using demeaned data (Within estimator)
    Y_stacked = [];
    X_stacked = [];
    for i = 1:nUnits
        Yi = Y_panel{i};
        % Demean for fixed effects
        Yi_dm = Yi - mean(Yi, 1);

        [Ti, Ki] = size(Yi_dm);
        Y_eff = Yi_dm(lags+1:end, :);
        Xi = [];
        for l = 1:lags
            Xi = [Xi, Yi_dm(lags+1-l : end-l, :)];
        end
        Y_stacked = [Y_stacked; Y_eff];
        X_stacked = [X_stacked; Xi];
    end

    % Estimate shared dynamics
    beta_dynamics = (X_stacked' * X_stacked) \ (X_stacked' * Y_stacked);

    % Recover intercepts unit-by-unit
    intercepts = zeros(nUnits, K);
    for i = 1:nUnits
        Yi = Y_panel{i};
        Yi_lag_sum = zeros(T-lags, K);
        for l = 1:lags
            Yi_lag_sum = Yi_lag_sum + Yi(lags+1-l : end-l, :) * beta_dynamics((l-1)*K+1 : l*K, :);
        end
        intercepts(i, :) = mean(Yi(lags+1:end, :) - Yi_lag_sum, 1);
    end

    results.Coefficients = beta_dynamics;
    results.Intercepts = intercepts;
    results.Type = "fixed_effects";
end
end
