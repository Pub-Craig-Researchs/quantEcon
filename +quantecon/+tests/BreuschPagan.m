function results = BreuschPagan(y, X, opts)
%BREUSCHPAGAN Breusch-Pagan / Koenker Heteroskedasticity Test
%
%   Tests H0: homoskedasticity vs H1: Var(e_i) = f(z_i'alpha), using either
%   the original Breusch-Pagan (1979) or the studentized Koenker (1981) variant.
%
%   Usage:
%       res = quantecon.tests.BreuschPagan(y, X);
%       res = quantecon.tests.BreuschPagan(y, X, 'Studentize', false);
%
%   Inputs:
%       y - (n x 1) dependent variable.
%       X - (n x p) regressors (do NOT include intercept; added automatically).
%
%   Options:
%       'Studentize' - (logical) Use Koenker's studentized variant (default: true).
%                      true  = robust to non-normality.
%                      false = original BP (requires normality).
%
%   Output:
%       results - Struct with fields:
%           .Stat   - Test statistic (Chi-squared).
%           .Pval   - P-value.
%           .df     - Degrees of freedom (= p).
%           .Type   - "Koenker" or "BreuschPagan".
%
%   Reference:
%       Breusch, T.S. & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity
%       and Random Coefficient Variation", Econometrica, 47(5), 1287-1294.
%       Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity",
%       Journal of Econometrics, 17(1), 107-112.

arguments
    y (:,1) double {mustBeNonempty}
    X (:,:) double {mustBeNonempty}
    opts.Studentize (1,1) logical = true
end

n = length(y);
p = size(X, 2);
assert(size(X, 1) == n, 'X and y must have the same number of rows.');

% --- OLS regression: y = [1, X] * beta + e ---
X_aug = [ones(n, 1), X];
b = X_aug \ y;
r = y - X_aug * b;

% --- Auxiliary regression: r^2 = [1, X] * gamma + v ---
r2 = r.^2;
b_aux = X_aug \ r2;
r2_hat = X_aug * b_aux;
RSS_aux = sum((r2 - r2_hat).^2);
TSS_aux = sum((r2 - mean(r2)).^2);
R2_aux = 1 - RSS_aux / TSS_aux;

% Test statistic: n * R^2 from auxiliary regression
stat = n * R2_aux;

% Koenker studentized correction
if ~opts.Studentize
    % Original BP: scale by 2*sigma^4 / (n-1)/n * var(r^2)
    sigma2 = (n - 1) / n * var(r);
    lam = (n - 1) / n * var(r2) / (2 * sigma2^2);
    stat = stat * lam;
end

pval = 1 - chi2cdf(abs(stat), p);

results.Stat = stat;
results.Pval = pval;
results.df = p;
results.Type = string(ternary(opts.Studentize, "Koenker", "BreuschPagan"));

end


function out = ternary(cond, a, b)
%TERNARY Inline conditional.
if cond
    out = a;
else
    out = b;
end
end
