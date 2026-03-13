function result = Denton(Y, x, opts)
% Denton  Temporal disaggregation via the Denton (1971) method
%
%   Non-parametric method that minimises the volatility of d-th
%   differences of the adjustment (y - x).  Supports additive and
%   proportional variants with first or second differencing.
%
% SYNTAX:
%   result = quantecon.timeseries.disagg.Denton(Y, x)
%   result = quantecon.timeseries.disagg.Denton(Y, x, Variant="proportional")
%
% INPUT:
%   Y  - (N x 1) low-frequency series
%   x  - (n x 1) high-frequency indicator (single column)
%
% OPTIONS:
%   AggType  - "sum" | "avg" | "last" | "first"  (default "sum")
%   Freq     - frequency ratio sc  (default 4)
%   DiffOrder- 1 or 2  (default 1, minimise d-th difference volatility)
%   Variant  - "additive" | "proportional"  (default "additive")
%
% OUTPUT:
%   result struct with fields:
%     .y  - (n x 1) disaggregated high-frequency estimate
%     .u  - (n x 1) high-frequency residuals (y - x)
%     .U  - (N x 1) low-frequency residuals (Y - C*x)
%
% REFERENCE:
%   Denton, F.T. (1971) "Adjustment of monthly or quarterly series
%   to annual totals: an approach based on quadratic minimization",
%   J. American Statistical Association, 66(333), 99-102.
%
% Source: Refactored from EMQuilis Temporal Disaggregation Toolbox v3.0.

arguments
    Y (:,1) double
    x (:,1) double
    opts.AggType (1,1) string {mustBeMember(opts.AggType,["sum","avg","last","first"])} = "sum"
    opts.Freq (1,1) double {mustBePositive,mustBeInteger} = 4
    opts.DiffOrder (1,1) double {mustBeMember(opts.DiffOrder,[1 2])} = 1
    opts.Variant (1,1) string {mustBeMember(opts.Variant,["additive","proportional"])} = "additive"
end

ta = find(opts.AggType == ["sum","avg","last","first"]);
sc = opts.Freq;
d  = opts.DiffOrder;
N  = length(Y);
n  = length(x);

% Aggregation matrix
C = aggreg_mat(ta, N, sc);
if n > sc * N
    C = [C, zeros(N, n - sc*N)];
end

% Low-frequency residuals
X = C * x;
U = Y - X;

% Difference matrix
D = dif_mat(d, n);

% Quadratic penalty matrix Q
if opts.Variant == "additive"
    % [FIX]: backslash instead of inv()
    Q = (D' * D) \ eye(n);
else
    % Proportional: Q = diag(x) * inv(D'*D) * diag(x)
    Q = diag(x) * ((D' * D) \ eye(n)) * diag(x);
end

% High-frequency residuals  u = Q*C' * inv(C*Q*C') * U
% [FIX]: use backslash for inner inverse
u = Q * C' * ((C * Q * C') \ U);

% Disaggregated series
y = x + u;

result = struct('y', y, 'u', u, 'U', U);
end

% =====================================================================

function D = dif_mat(d, n)
% Generate n x n difference operator matrix (order d, zero initial conds)
    switch d
        case 1
            D = eye(n) + diag(-ones(n-1,1), -1);
        case 2
            D = eye(n) + diag(-2*ones(n-1,1), -1) + diag(ones(n-2,1), -2);
    end
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
