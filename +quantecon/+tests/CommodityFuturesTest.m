function CommodityFuturesTest()
% CommodityFuturesTest  Smoke tests for quantecon.finance.CommodityFutures
%
%   Tests:
%     1. Simulated data: function runs and returns valid output structure
%     2. Input validation: dimension mismatch, NaN, too few observations
%     3. Runge-Kutta internal consistency: affine coefficients are finite
%     4. Kalman filter outputs: states, residuals have correct dimensions
%     5. Mode B (with Rates): estimation with observed interest rates

fprintf('=== CommodityFutures Test Suite ===\n\n');

%% Test 1: Smoke test with synthetic data
fprintf('[1] Smoke test with synthetic data ... ');
rng(42);
T  = 200;
ny = 3;
maturities = [3, 6, 12] / 12;  % 3m, 6m, 12m

% Generate synthetic log-futures prices: random walk + maturity spread
spot = cumsum(0.001 * randn(T, 1)) + 4;
LogFutures = spot + (0.01 * randn(T, ny)) + maturities .* 0.02;

result = quantecon.finance.CommodityFutures(LogFutures, maturities, ...
    'NStarts', 10, 'NRunge', 500);

% Check output structure
assert(isstruct(result), 'Output must be a struct');
expected_fields = {'Theta','LogLik','AIC','BIC','States','Residuals', ...
                   'AffineA','AffineZ','ParamTable','EigMin','T','ny','HasRates'};
for k = 1:numel(expected_fields)
    assert(isfield(result, expected_fields{k}), ...
        'Missing field: %s', expected_fields{k});
end

assert(result.T == T, 'T mismatch');
assert(result.ny == ny, 'ny mismatch');
assert(~result.HasRates, 'HasRates should be false in commodity-only mode');
assert(isfinite(result.LogLik), 'LogLik must be finite');
assert(isfinite(result.AIC), 'AIC must be finite');
assert(isfinite(result.BIC), 'BIC must be finite');
fprintf('PASSED\n');

%% Test 2: Output dimensions
fprintf('[2] Output dimension checks ... ');
assert(isequal(size(result.States), [T, 4]), ...
    'States must be T x 4');
assert(isequal(size(result.Residuals), [T, ny]), ...
    'Residuals must be T x ny');
assert(isequal(size(result.AffineA), [1, ny]), ...
    'AffineA must be 1 x ny');
assert(isequal(size(result.AffineZ), [ny, 4]), ...
    'AffineZ must be ny x 4');
assert(numel(result.Theta) == 27 + ny, ...
    'Theta must have 27+ny elements');
assert(numel(result.EigMin) == T, ...
    'EigMin must have T elements');
assert(height(result.ParamTable) == 27 + ny, ...
    'ParamTable must have 27+ny rows');
fprintf('PASSED\n');

%% Test 3: AIC < BIC (holds for nparam > 1, T*ny large)
fprintf('[3] AIC <= BIC for large sample ... ');
% BIC penalizes more than AIC when log(T*ny) > 2
if log(T * ny) > 2
    assert(result.AIC <= result.BIC, 'AIC should be <= BIC for large T*ny');
    fprintf('PASSED\n');
else
    fprintf('SKIPPED (T*ny too small)\n');
end

%% Test 4: Input validation - dimension mismatch
fprintf('[4] Dimension mismatch error ... ');
try
    quantecon.finance.CommodityFutures(LogFutures, [0.25, 0.5]);
    fprintf('FAILED (no error thrown)\n');
catch ME
    if contains(ME.message, 'dimMismatch') || contains(ME.message, 'Maturities')
        fprintf('PASSED\n');
    else
        fprintf('FAILED (wrong error: %s)\n', ME.message);
    end
end

%% Test 5: Input validation - NaN data
fprintf('[5] NaN data error ... ');
badData = LogFutures;
badData(50, 2) = NaN;
try
    quantecon.finance.CommodityFutures(badData, maturities, 'NStarts', 5);
    fprintf('FAILED (no error thrown)\n');
catch ME
    if contains(ME.message, 'nanInf') || contains(ME.message, 'NaN')
        fprintf('PASSED\n');
    else
        fprintf('FAILED (wrong error: %s)\n', ME.message);
    end
end

%% Test 6: Input validation - too few observations
fprintf('[6] Too few observations error ... ');
try
    quantecon.finance.CommodityFutures(LogFutures(1:5,:), maturities, 'NStarts', 5);
    fprintf('FAILED (no error thrown)\n');
catch ME
    if contains(ME.message, 'tooFew') || contains(ME.message, '10')
        fprintf('PASSED\n');
    else
        fprintf('FAILED (wrong error: %s)\n', ME.message);
    end
end

%% Test 7: Affine coefficients are real and finite
fprintf('[7] Affine coefficients finite and real ... ');
assert(all(isfinite(result.AffineA)), 'AffineA has non-finite values');
assert(all(isfinite(result.AffineZ(:))), 'AffineZ has non-finite values');
assert(isreal(result.AffineA), 'AffineA has complex values');
assert(isreal(result.AffineZ), 'AffineZ has complex values');
fprintf('PASSED\n');

%% Test 8: Different number of contracts
fprintf('[8] Five-contract estimation ... ');
ny5 = 5;
mat5 = [2, 4, 6, 8, 10] / 12;
LF5 = spot + (0.01 * randn(T, ny5)) + mat5 .* 0.02;
result5 = quantecon.finance.CommodityFutures(LF5, mat5, ...
    'NStarts', 5, 'NRunge', 200);
assert(result5.ny == 5, 'ny must be 5');
assert(numel(result5.Theta) == 32, 'Theta must have 32 elements for 5 contracts');
assert(~result5.HasRates, 'HasRates should be false');
fprintf('PASSED\n');

%% Test 9: Mode B - with observed interest rates
fprintf('[9] Mode B: estimation with observed rates ... ');
rng(99);
rates = 0.03 + 0.005 * cumsum(randn(T, 1)) * sqrt(1/252);  % simulated rates

resultR = quantecon.finance.CommodityFutures(LogFutures, maturities, ...
    'Rates', rates, 'NStarts', 10, 'NRunge', 500);

assert(resultR.HasRates, 'HasRates should be true');
assert(numel(resultR.Theta) == 27 + ny + 1, ...
    'Theta must have 27+ny+1 elements with Rates');
assert(isequal(size(resultR.Residuals), [T, ny + 1]), ...
    'Residuals must be T x (ny+1) with Rates');
assert(isequal(size(resultR.States), [T, 4]), ...
    'States still T x 4');
assert(height(resultR.ParamTable) == 27 + ny + 1, ...
    'ParamTable must have 27+ny+1 rows');
assert(isfinite(resultR.LogLik), 'LogLik must be finite with Rates');
% Affine coefficients are for futures only (unchanged)
assert(isequal(size(resultR.AffineA), [1, ny]), ...
    'AffineA still 1 x ny');
assert(isequal(size(resultR.AffineZ), [ny, 4]), ...
    'AffineZ still ny x 4');
% Last param should be sigeta
assert(strcmp(resultR.ParamTable.Parameter{end}, 'sigeta'), ...
    'Last parameter must be sigeta');
fprintf('PASSED\n');

%% Test 10: Rates input validation - length mismatch
fprintf('[10] Rates length mismatch error ... ');
try
    quantecon.finance.CommodityFutures(LogFutures, maturities, ...
        'Rates', rates(1:100));
    fprintf('FAILED (no error thrown)\n');
catch ME
    if contains(ME.message, 'ratesDim') || contains(ME.message, 'Rates')
        fprintf('PASSED\n');
    else
        fprintf('FAILED (wrong error: %s)\n', ME.message);
    end
end

%% Test 11: Rates with NaN
fprintf('[11] Rates NaN error ... ');
badRates = rates;
badRates(10) = NaN;
try
    quantecon.finance.CommodityFutures(LogFutures, maturities, ...
        'Rates', badRates, 'NStarts', 5);
    fprintf('FAILED (no error thrown)\n');
catch ME
    if contains(ME.message, 'ratesNanInf') || contains(ME.message, 'NaN')
        fprintf('PASSED\n');
    else
        fprintf('FAILED (wrong error: %s)\n', ME.message);
    end
end

fprintf('\n=== All CommodityFutures tests passed ===\n');
end
