function results = Midas(y, X_hf, V_lf, opts)
%MIDAS Mixed Data Sampling Regression (Functional Entry Point)
%
%   Usage:
%       results = quantecon.midas.Midas(y, X_hf)
%       results = quantecon.midas.Midas(y, X_hf, V_lf, 'Polynomial', 'Beta')
%
%   Inputs:
%       y    - (T x 1) Dependent variable (low frequency)
%       X_hf - (T x p) High-frequency regressors (lagged format)
%       V_lf - (T x k) Low-frequency regressors (optional, default: intercept)
%
%   Options:
%       Polynomial - Type of weighting ('Beta', 'ExpAlmon', 'Almon')
%       Lags       - Number of HF lags (default: size(X_hf, 2))
%       BetaCase   - For Beta polynomial: 1 (theta1=1), 2 (theta3=0), 3 (both), 0 (neither)
%       MultiStart - (logical) Use MultiStart for non-linear optimization
%
%   References:
%       Audrino, Kostrov, Ortega (2019), Kostrov (2020)

arguments
    y (:,1) double {mustBeNumeric, mustBeReal}
    X_hf (:,:) double {mustBeNumeric, mustBeReal}
    V_lf (:,:) double = []
    opts.Polynomial (1,1) string {mustBeMember(opts.Polynomial, ["Beta", "ExpAlmon", "Almon"])} = "Beta"
    opts.Lags (1,1) double = size(X_hf, 2)
    opts.BetaCase (1,1) double {mustBeMember(opts.BetaCase, [0, 1, 2, 3])} = 3
    opts.MultiStart (1,1) logical = false
    opts.Runs (1,1) double = 16
end

T = size(y, 1);
if size(X_hf, 1) ~= T
    error("quantecon:midas:Midas:DimensionMismatch", "Row of X_hf must match y.");
end

% Default low-frequency regressors: Intercept only if V_lf is empty
if isempty(V_lf)
    V_lf = ones(T, 1);
else
    % Check if V_lf already contains a constant
    if ~any(all(V_lf == 1, 1))
        V_lf = [ones(T, 1), V_lf];
    end
end

% Call estimation engine
results = quantecon.midas.estimate_midas(y, X_hf, V_lf, opts);

end
