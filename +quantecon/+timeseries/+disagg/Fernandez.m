function result = Fernandez(Y, x, opts)
% Fernandez  Temporal disaggregation via the Fernandez (1981) method
%
%   Special case of Litterman with rho = 1 (unit root in residuals).
%   No rho optimisation required — closed-form GLS solution.
%   Error structure: (I + L) e_t = a_t  (random walk).
%
% SYNTAX:
%   result = quantecon.timeseries.disagg.Fernandez(Y, x)
%   result = quantecon.timeseries.disagg.Fernandez(Y, x, Freq=12)
%
% INPUT:
%   Y  - (N x 1) low-frequency series
%   x  - (n x p) high-frequency indicator matrix (without intercept)
%
% OPTIONS:
%   AggType   - "sum" | "avg" | "last" | "first"  (default "sum")
%   Freq      - frequency ratio sc  (default 4)
%   Intercept - "auto" | "none" | "include"  (default "auto")
%
% OUTPUT:
%   result struct: .y, .y_sd, .y_lo, .y_up, .beta, .beta_sd, .beta_t,
%                  .rho (=1), .sigma, .aic, .bic, .u, .U
%
% REFERENCE:
%   Fernandez, R.B. (1981) "Methodological note on the estimation of
%   time series", Review of Economics and Statistics, 63(3), 471-478.
%
% Source: Refactored from EMQuilis Temporal Disaggregation Toolbox v1.1.

arguments
    Y (:,1) double
    x (:,:) double
    opts.AggType (1,1) string {mustBeMember(opts.AggType,["sum","avg","last","first"])} = "sum"
    opts.Freq (1,1) double {mustBePositive,mustBeInteger} = 4
    opts.Intercept (1,1) string {mustBeMember(opts.Intercept,["auto","none","include"])} = "auto"
end

ta = find(opts.AggType == ["sum","avg","last","first"]);
sc = opts.Freq;
N  = length(Y);
[n, p] = size(x);

% --- Intercept pretest ---
if opts.Intercept == "auto"
    pre = fern_core(Y, x, ta, sc, true, N, n, p);
    addInt = abs(pre.beta_t(1)) >= 2;
elseif opts.Intercept == "include"
    addInt = true;
else
    addInt = false;
end

res = fern_core(Y, x, ta, sc, addInt, N, n, p);

result = struct( ...
    'y',       res.y,       'y_sd',    res.y_sd, ...
    'y_lo',    res.y_lo,    'y_up',    res.y_up, ...
    'beta',    res.beta,    'beta_sd', res.beta_sd, ...
    'beta_t',  res.beta_t,  'rho',     1.0, ...
    'sigma',   res.sigma,   'aic',     res.aic, ...
    'bic',     res.bic,     'u',       res.u, ...
    'U',       res.U);
end

% =====================================================================

function res = fern_core(Y, x, ta, sc, addInt, N, n, p)
    if addInt, x = [ones(n,1), x]; p = p + 1; end
    C = aggreg_mat(ta, N, sc);
    if n > sc * N, C = [C, zeros(N, n - sc*N)]; end
    X = C * x;

    % Fernandez: Aux = I + LL  (rho = 1, no AR parameter)
    In = eye(n);
    LL = diag(-ones(n-1,1), -1);
    Aux = In + LL;
    % [FIX]: backslash instead of inv()
    w  = (Aux' * Aux) \ In;
    W  = C * w * C';
    Wi = W \ eye(N);

    beta = (X' * Wi * X) \ (X' * Wi * Y);
    U    = Y - X * beta;
    wls  = U' * Wi * U;
    sig  = wls / (N - p);

    L = w * C' * Wi;
    u = L * U;
    y = x * beta + u;

    % Information criteria (no rho parameter, so just p)
    res.aic = log(sig) + 2*p/N;
    res.bic = log(sig) + log(N)*p/N;

    sig_beta = sig * ((X' * Wi * X) \ eye(p));
    VCV1 = sig * (In - L*C) * w;
    VCV2 = (x - L*X) * sig_beta * (x - L*X)';
    d_y  = sqrt(abs(diag(VCV1 + VCV2)));

    res.y       = y;
    res.y_sd    = d_y;
    res.y_lo    = y - d_y;
    res.y_up    = y + d_y;
    res.beta    = beta;
    res.beta_sd = sqrt(diag(sig_beta));
    res.beta_t  = beta ./ res.beta_sd;
    res.sigma   = sig;
    res.u       = u;
    res.U       = U;
end

function C = aggreg_mat(ta, N, sc)
    switch ta
        case 1, c = ones(1,sc);
        case 2, c = ones(1,sc)/sc;
        case 3, c = [zeros(1,sc-1), 1];
        case 4, c = [1, zeros(1,sc-1)];
    end
    C = kron(eye(N), c);
end
