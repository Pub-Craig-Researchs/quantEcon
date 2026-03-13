function results = AssetIndicators(type, y, X_factors, opts)
%ASSETINDICATORS Collection of Asset Pricing Indicators
%
%   Usage:
%       val = quantecon.finance.AssetIndicators("amihud", returns, dollar_volume)
%       val = quantecon.finance.AssetIndicators("ivol", returns, factors)
%       val = quantecon.finance.AssetIndicators("coskewness", returns, market_returns)
%
%   Types:
%       "amihud"     - Illiquidity measure (Amihud 2002)
%       "ivol"       - Idiosyncratic Volatility (Ang et al. 2006)
%       "coskewness" - Systemic Skewness (Harvey & Siddique 2000)
%       "momentum"   - Cumulative returns over a window.

arguments
    type (1,1) string {mustBeMember(type, ["amihud", "ivol", "coskewness", "momentum"])}
    y (:,1) double {mustBeNumeric, mustBeReal}
    X_factors (:,:) double = []
    opts.Window (1,1) double = length(y)
    opts.MinObs (1,1) double = 15
end

% Filter NaNs
valid_idx = ~isnan(y);
if ~isempty(X_factors)
    valid_idx = valid_idx & all(~isnan(X_factors), 2);
end

y = y(valid_idx);
if ~isempty(X_factors)
    X_factors = X_factors(valid_idx, :);
end

if length(y) < opts.MinObs
    results = NaN;
    return;
end

switch type
    case "amihud"
        % y = returns, X_factors = dollar_volume
        if isempty(X_factors)
            error("Amihud requires dollar volume as second argument.");
        end
        results = mean(abs(y) ./ X_factors, "omitnan") * 1e9;

    case "ivol"
        % y = excess returns, X_factors = [Mkt-RF, SMB, HML]
        if isempty(X_factors)
            error("ivol requires factor matrix as second argument.");
        end
        % Use OLS from +base
        reg = quantecon.base.Ols(y, X_factors, 'HasConstant', true);
        % IVOL is stdev of residuals (annualized)
        results = std(reg.Residuals) * sqrt(252);

    case "coskewness"
        % y = excess returns, X_factors = market excess returns
        if isempty(X_factors)
            error("coskewness requires market returns as second argument.");
        end
        % Harvey & Siddique (2000)
        % Regress R_i on R_m and R_m^2
        X_design = [X_factors, X_factors.^2];
        reg = quantecon.base.Ols(y, X_design, 'HasConstant', true);

        eps_i = reg.Residuals;
        eps_m = X_factors - mean(X_factors);

        numerator = mean(eps_i .* (eps_m.^2));
        denominator = sqrt(mean(eps_i.^2)) * mean(eps_m.^2);
        results = numerator / denominator;

    case "momentum"
        % Cumulative return over the window
        % y = returns
        results = prod(1 + y) - 1;
end

end
