function result = ChowLin(Y, x, opts)
% ChowLin  Temporal disaggregation via the Chow-Lin (1971) method
%
%   Distributes low-frequency data (e.g., annual) to high-frequency
%   (e.g., quarterly) using related high-frequency indicator series.
%   Residuals follow an AR(1) process; rho estimated by grid search.
%
% SYNTAX:
%   result = quantecon.timeseries.disagg.ChowLin(Y, x)
%   result = quantecon.timeseries.disagg.ChowLin(Y, x, AggType="sum", Freq=4)
%
% INPUT:
%   Y  - (N x 1) low-frequency series
%   x  - (n x p) high-frequency indicator matrix (without intercept)
%
% OPTIONS:
%   AggType   - "sum" | "avg" | "last" | "first"  (default "sum")
%   Freq      - frequency ratio sc  (default 4, annual->quarterly)
%   Method    - "ml" | "wls"  (default "ml")
%   Intercept - "auto" | "none" | "include"  (default "auto")
%   Rho       - [] = default grid | scalar = fixed | [rmin rmax ngrid]
%
% OUTPUT:
%   result struct with fields:
%     .y, .y_sd, .y_lo, .y_up  - disaggregated series and confidence band
%     .beta, .beta_sd, .beta_t - regression coefficients
%     .rho     - estimated AR(1) parameter
%     .sigma   - innovation variance
%     .aic, .bic - information criteria
%     .u, .U   - high / low frequency residuals
%
% REFERENCE:
%   Chow, G. and Lin, A.L. (1971) "Best linear unbiased distribution
%   and extrapolation of economic time series by related series",
%   Review of Economics and Statistics, 53(4), 372-375.
%
% Source: Refactored from EMQuilis Temporal Disaggregation Toolbox v3.4.
%         All inv() calls replaced with backslash.

arguments
    Y (:,1) double
    x (:,:) double
    opts.AggType (1,1) string {mustBeMember(opts.AggType,["sum","avg","last","first"])} = "sum"
    opts.Freq (1,1) double {mustBePositive,mustBeInteger} = 4
    opts.Method (1,1) string {mustBeMember(opts.Method,["ml","wls"])} = "ml"
    opts.Intercept (1,1) string {mustBeMember(opts.Intercept,["auto","none","include"])} = "auto"
    opts.Rho double = []
end

ta   = find(opts.AggType == ["sum","avg","last","first"]);
sc   = opts.Freq;
type = double(opts.Method == "ml");   % 0=wls, 1=ml
N    = length(Y);
[n, p] = size(x);

% --- Intercept pretest (opC = -1 logic) ---
if opts.Intercept == "auto"
    pre = cl_core(Y, x, ta, sc, type, true, opts.Rho, N, n, p);
    addInt = abs(pre.beta_t(1)) >= 2;
elseif opts.Intercept == "include"
    addInt = true;
else
    addInt = false;
end

res = cl_core(Y, x, ta, sc, type, addInt, opts.Rho, N, n, p);

result = struct( ...
    'y',       res.y,       'y_sd',    res.y_sd, ...
    'y_lo',    res.y_lo,    'y_up',    res.y_up, ...
    'beta',    res.beta,    'beta_sd', res.beta_sd, ...
    'beta_t',  res.beta_t,  'rho',     res.rho, ...
    'sigma',   res.sigma,   'aic',     res.aic, ...
    'bic',     res.bic,     'u',       res.u, ...
    'U',       res.U);
end

% =====================================================================
% LOCAL FUNCTIONS
% =====================================================================

function res = cl_core(Y, x, ta, sc, type, addInt, rl, N, n, p)
    if addInt
        x = [ones(n,1), x];
        p = p + 1;
    end
    C = aggreg_mat(ta, N, sc);
    if n > sc * N
        C = [C, zeros(N, n - sc*N)];
    end
    X = C * x;

    % --- Optimize rho ---
    nrl = numel(rl);
    if nrl == 0
        rho = grid_rho_cl(Y, X, C, N, n, type, [0.05 0.99 50]);
    elseif nrl == 1
        rho = rl;
    elseif nrl == 3
        rho = grid_rho_cl(Y, X, C, N, n, type, rl);
    else
        error('ChowLin:BadRho','Rho must be [], scalar, or [rmin rmax ngrid].');
    end

    % --- GLS with optimal rho ---
    In = eye(n);
    LL = diag(-ones(n-1,1), -1);
    Aux = In + rho * LL;
    Aux(1,1) = sqrt(1 - rho^2);
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

    % Information criteria (p+1 includes rho)
    res.aic = log(sig) + 2*(p+1)/N;
    res.bic = log(sig) + log(N)*(p+1)/N;

    % VCV of estimates
    sig_beta = sig * ((X' * Wi * X) \ eye(p));
    VCV1 = sig * (In - L*C) * w;
    VCV2 = (x - L*X) * sig_beta * (x - L*X)';
    d_y  = sqrt(abs(diag(VCV1 + VCV2)));  % [FIX]: abs guards numerical noise

    res.y       = y;
    res.y_sd    = d_y;
    res.y_lo    = y - d_y;
    res.y_up    = y + d_y;
    res.beta    = beta;
    res.beta_sd = sqrt(diag(sig_beta));
    res.beta_t  = beta ./ res.beta_sd;
    res.rho     = rho;
    res.sigma   = sig;
    res.u       = u;
    res.U       = U;
end

function rho = grid_rho_cl(Y, X, C, N, n, type, rl)
% Grid search for optimal rho — Chow-Lin AR(1) criterion
    r  = linspace(rl(1), rl(2), rl(3));
    nr = length(r);
    In = eye(n);
    LL = diag(-ones(n-1,1), -1);
    val = zeros(1, nr);
    for h = 1:nr
        Aux = In + r(h) * LL;
        Aux(1,1) = sqrt(1 - r(h)^2);
        w  = (Aux' * Aux) \ In;
        W  = C * w * C';
        Wi = W \ eye(N);
        b  = (X' * Wi * X) \ (X' * Wi * Y);
        Uh = Y - X * b;
        wh = Uh' * Wi * Uh;
        sh = wh / N;
        lh = (-N/2)*log(2*pi*sh) - 0.5*log(det(W)) - N/2;
        val(h) = (1 - type)*(-wh) + type * lh;
    end
    [~, idx] = max(val);
    rho = r(idx);
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
