function result = HistVol(data, opts)
% HistVol  Historical volatility estimators (9 methods)
%
%   Computes annualised rolling-window volatility from OHLC price data.
%   Supports: Close-to-Close (CC/CCD), Parkinson, Garman-Klass,
%   Rogers-Satchell, Yang-Zhang, GKYZ, Hodges-Tompkins, Meilijson.
%
% SYNTAX:
%   result = quantecon.finance.HistVol(data)
%   result = quantecon.finance.HistVol(data, Method="YZ", Window=21)
%
% INPUT:
%   data - (T x 4) matrix [Open, High, Low, Close]  (OHLC prices)
%          or (T x 1) vector of close prices (only CC, CCD, HT available)
%
% OPTIONS:
%   Method   - estimator: "CC"|"CCD"|"GK"|"GKYZ"|"HT"|"M"|"P"|"RS"|"YZ"
%              (default "CC")
%   Window   - rolling window size [2, T-1]  (default 21)
%   AnnFactor- annualisation factor  (default 252)
%   Clean    - logical, remove leading NaN  (default false)
%
% OUTPUT:
%   result struct with fields:
%     .vol    - (T x 1) annualised volatility series (NaN for initial obs)
%     .method - string label
%
% ESTIMATOR KEY:
%   CC   - Close-to-Close (standard)
%   CCD  - Close-to-Close demeaned
%   GK   - Garman & Klass (1980)
%   GKYZ - Garman-Klass + Yang-Zhang (2000) overnight extension
%   HT   - Hodges & Tompkins (2002) bias-corrected
%   M    - Meilijson (2009) composite
%   P    - Parkinson (1980) high-low range
%   RS   - Rogers & Satchell (1991) drift-independent
%   YZ   - Yang & Zhang (2000) optimal composite
%
% Source: Refactored from TommasoBelluzzo HistoricalVolatility toolbox.

arguments
    data (:,:) double
    opts.Method (1,1) string {mustBeMember(opts.Method,...
        ["CC","CCD","GK","GKYZ","HT","M","P","RS","YZ"])} = "CC"
    opts.Window (1,1) double {mustBePositive, mustBeInteger} = 21
    opts.AnnFactor (1,1) double {mustBePositive} = 252
    opts.Clean (1,1) logical = false
end

[T, nc] = size(data);
bw  = opts.Window;
ann = opts.AnnFactor;

% Parse OHLC
if nc == 4
    O = data(:,1); H = data(:,2); L = data(:,3); C = data(:,4);
    ret = [NaN; diff(log(C))];
elseif nc == 1
    C = data;
    ret = [NaN; diff(log(C))];
    O = []; H = []; L = [];
else
    error('HistVol:BadInput','data must be T x 4 (OHLC) or T x 1 (Close).');
end

% Check OHLC requirement
needOHLC = ismember(opts.Method, ["GK","GKYZ","M","P","RS","YZ"]);
if needOHLC && isempty(O)
    error('HistVol:NeedOHLC','Method "%s" requires OHLC data (T x 4).', opts.Method);
end

% Compute per-period quantities
switch opts.Method
    case "CC"
        q = ret.^2;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / (bw - 1));

    case "CCD"
        mu = mean(ret, 'omitnan');
        q = (ret - mu).^2;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / (bw - 1));

    case "P"
        q = log(H ./ L).^2;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / (bw * 4 * log(2)));

    case "GK"
        co = log(C ./ O);
        hl = log(H ./ L);
        q = 0.5 * hl.^2 - (2*log(2) - 1) * co.^2;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / bw);

    case "RS"
        co = log(C ./ O);
        ho = log(H ./ O);
        lo = log(L ./ O);
        q = ho .* (ho - co) + lo .* (lo - co);
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / bw);

    case "GKYZ"
        co = log(C ./ O);
        hl = log(H ./ L);
        oc = [NaN; log(O(2:end) ./ C(1:end-1))].^2;
        q = oc + 0.5 * hl.^2 - (2*log(2) - 1) * co.^2;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / bw);

    case "HT"
        dif = T - bw;
        param = sqrt(ann / (1 - bw/dif + (bw^2 - 1)/(3*dif^2)));
        vol = rolling_std(ret, bw) * param;

    case "M"
        co = log(C ./ O);
        ho = log(H ./ O);
        lo = log(L ./ O);
        co_neg = co < 0;
        co_swi = -co;
        ho_swi = ho; ho_swi(co_neg) = -lo(co_neg);
        lo_swi = lo; lo_swi(co_neg) = -ho(co_neg);
        s1 = 2 * ((ho_swi - co_swi).^2 + lo_swi.^2);
        s2 = co.^2;
        s3 = 2 * (ho_swi - co_swi - lo_swi) .* co_swi;
        s4 = -((ho_swi - co_swi) .* lo_swi) / (2*log(2) - 1.25);
        q = 0.273520*s1 + 0.160358*s2 + 0.365212*s3 + 0.200910*s4;
        vol = rolling_sqrt_sum(q, bw) * sqrt(ann / bw);

    case "YZ"
        co = log(C ./ O);
        ho = log(H ./ O);
        lo = log(L ./ O);
        oc2 = [NaN; log(O(2:end) ./ C(1:end-1))].^2;
        co2 = co.^2;
        rs  = ho .* (ho - co) + lo .* (lo - co);
        k   = 0.34 / (1.34 + (bw + 1)/(bw - 1));
        % Yang-Zhang: sigma^2 = sigma_oc^2 + k*sigma_cc^2 + (1-k)*sigma_rs^2
        s_oc = rolling_mean(oc2, bw) * ann;
        s_cc = rolling_mean(co2, bw) * ann;
        s_rs = rolling_mean(rs, bw) * ann;
        vol = sqrt(s_oc + k * s_cc + (1-k) * s_rs);
end

if opts.Clean
    vol = vol(~isnan(vol));
end

result = struct('vol', vol, 'method', opts.Method);
end

% =====================================================================

function v = rolling_sqrt_sum(q, bw)
% sqrt of rolling sum
    T = length(q);
    v = NaN(T, 1);
    for t = bw:T
        v(t) = sqrt(sum(q(t-bw+1:t), 'omitnan'));
    end
end

function v = rolling_std(q, bw)
% Rolling standard deviation
    T = length(q);
    v = NaN(T, 1);
    for t = bw:T
        v(t) = std(q(t-bw+1:t), 0, 'omitnan');
    end
end

function v = rolling_mean(q, bw)
% Rolling mean
    T = length(q);
    v = NaN(T, 1);
    for t = bw:T
        v(t) = mean(q(t-bw+1:t), 'omitnan');
    end
end
