function indicators = TechnicalIndicators(price, volume, opts)
%TECHNICALINDICATORS Compute standard financial technical indicators
%
%   Usage:
%       res = quantecon.finance.TechnicalIndicators(price, volume)
%
%   Inputs:
%       price  - (T x 1) Closing prices
%       volume - (T x 1) Volume data (Optional, required for OBV)
%
%   Options:
%       'MA_Short'  - (int vector) Short MA windows (Default: [1 2 3])
%       'MA_Long'   - (int vector) Long MA windows (Default: [9 12])
%       'MOM_Lags'  - (int vector) Momentum lags (Default: [1 2 3 6 9 12])
%
%   Outputs:
%       indicators - Struct with MA, MOM, and OBV indicators.

arguments
    price (:,1) double {mustBeNumeric, mustBeReal}
    volume (:,1) double = []
    opts.MA_Short (1,:) double = [1 2 3]
    opts.MA_Long (1,:) double = [9 12]
    opts.MOM_Lags (1,:) double = [1 2 3 6 9 12]
end

T = length(price);
indicators = struct();

%% 1. Moving Average (MA) Binary Indicators
% Indicates if short-term mean > long-term mean
ma_results = [];
ma_names = [];
for s = opts.MA_Short
    for l = opts.MA_Long
        if s >= l, continue; end
        ma_s = movmean(price, [s-1 0]);
        ma_l = movmean(price, [l-1 0]);

        % Set signal to 1 if short > long
        signal = double(ma_s > ma_l);
        ma_results = [ma_results, signal];
        ma_names = [ma_names; string(sprintf('MA_%d_%d', s, l))];
    end
end
indicators.MA = ma_results;
indicators.MA_Names = ma_names;

%% 2. Momentum (MOM) Binary Models
% Indicates if current price > price K periods ago
mom_results = [];
mom_names = [];
for k = opts.MOM_Lags
    signal = double(price > [nan(k, 1); price(1:T-k)]);
    mom_results = [mom_results, signal];
    mom_names = [mom_names; string(sprintf('MOM_%d', k))];
end
indicators.MOM = mom_results;
indicators.MOM_Names = mom_names;

%% 3. On-Balance Volume (OBV)
if ~isempty(volume)
    price_diff = [0; diff(price)];
    obv = zeros(T, 1);
    for t = 2:T
        if price_diff(t) > 0
            obv(t) = volume(t);
        elseif price_diff(t) < 0
            obv(t) = -volume(t);
        end
    end
    indicators.OBV_Raw = cumsum(obv);

    % OBV Binary Signals (MA of OBV)
    obv_signals = [];
    for s = opts.MA_Short
        for l = opts.MA_Long
            if s >= l, continue; end
            ma_s = movmean(indicators.OBV_Raw, [s-1 0]);
            ma_l = movmean(indicators.OBV_Raw, [l-1 0]);
            obv_signals = [obv_signals, double(ma_s > ma_l)];
        end
    end
    indicators.OBV_Signals = obv_signals;
end
end
