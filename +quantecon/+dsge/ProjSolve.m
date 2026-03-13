function [POL, GRID, exitflag] = ProjSolve(fun_res, lb, ub, Y0, options)
%PROJSOLVE Solve DSGE models via projection methods (Chebyshev collocation).
%
%   Finds the policy function y = g(x) that satisfies the Euler equation
%   residuals: R(g; x) = 0 for all x on a Chebyshev grid.
%
%   Approximates the policy function using Chebyshev polynomials:
%       g(x) ~ sum_k theta_k * T_k(x_scaled)
%   where T_k are Chebyshev basis functions and x_scaled maps [lb, ub] to [-1, 1].
%
%   INPUTS:
%       fun_res - function handle: R = fun_res(Y, X, POL, GRID)
%                 R is (mm x dd) matrix of Euler residuals, where mm = number
%                 of grid points and dd = number of policy variables.
%                 The function must evaluate the policy at arbitrary points
%                 using the provided POL structure and evaluate_policy().
%       lb      - (1 x nn) lower bounds of state space
%       ub      - (1 x nn) upper bounds of state space
%       Y0      - (mm x dd) initial guess for policy on grid
%
%   Name-Value Options:
%       Order    - Chebyshev polynomial order per dimension.     Default: 5
%       NumNodes - grid points per dimension.                    Default: Order+1
%       MaxIter  - maximum time-iteration steps.                 Default: 500
%       Tol      - convergence tolerance.                        Default: 1e-8
%       Damping  - dampening in update (0 = none, 1 = full).    Default: 0
%       Display  - show iteration progress.                      Default: true
%
%   OUTPUTS:
%       POL (struct):
%           .theta  - (pp x dd) polynomial coefficients
%           .algo   - algorithm identifier
%       GRID (struct):
%           .nn     - number of state variables
%           .lb, .ub- bounds
%           .mm     - total grid points
%           .xx     - (mm x nn) grid in original coordinates
%           .xx_dw  - (mm x nn) grid scaled to [-1,1]
%           .XX_poly- (mm x pp) polynomial basis matrix
%       exitflag:
%           1 = converged, 0 = max iterations reached
%
%   Reference:
%       Judd, K.L. (1998). "Numerical Methods in Economics." MIT Press.
%       Promes Toolbox (Duineveld, 2021).
%
%   See also: quantecon.dsge.PertSolve, quantecon.dsge.Tauchen

arguments
    fun_res  function_handle
    lb       (1,:) double
    ub       (1,:) double
    Y0       (:,:) double
    options.Order    (1,1) double {mustBeInteger, mustBePositive} = 5
    options.NumNodes (1,1) double {mustBeInteger, mustBePositive} = 0
    options.MaxIter  (1,1) double {mustBeInteger, mustBePositive} = 500
    options.Tol      (1,1) double {mustBePositive} = 1e-8
    options.Damping  (1,1) double {mustBeInRange(options.Damping, 0, 1)} = 0
    options.Display  (1,1) logical = true
end

nn = length(lb);
ord = options.Order;
qq = options.NumNodes;
if qq == 0; qq = ord + 1; end

% ============================================================
% Step 1: Construct Chebyshev grid
% ============================================================
GRID.nn = nn;
GRID.lb = lb;
GRID.ub = ub;

% Chebyshev nodes per dimension (extrema of T_qq)
nodes_1d = cell(1, nn);
for d = 1:nn
    j = (1:qq)';
    nodes_1d{d} = -cos(pi * (j - 1) / (qq - 1));  % in [-1, 1]
end

% Tensor-product grid
if nn == 1
    xx_dw = nodes_1d{1};
else
    grids = cell(1, nn);
    [grids{:}] = ndgrid(nodes_1d{:});
    xx_dw = zeros(numel(grids{1}), nn);
    for d = 1:nn
        xx_dw(:, d) = grids{d}(:);
    end
end

mm = size(xx_dw, 1);
GRID.mm = mm;
GRID.xx_dw = xx_dw;

% Map to original coordinates
xx = zeros(mm, nn);
for d = 1:nn
    xx(:, d) = lb(d) + (xx_dw(:, d) + 1) / 2 * (ub(d) - lb(d));
end
GRID.xx = xx;

% Chebyshev polynomial basis (complete)
poly_elem = build_poly_elements(nn, ord);
pp = size(poly_elem, 1);
XX_poly = eval_cheb_basis(xx_dw, poly_elem);
GRID.XX_poly = XX_poly;
GRID.poly_elem = poly_elem;
GRID.ord = ord;

% ============================================================
% Step 2: Time iteration
% ============================================================
dd = size(Y0, 2);
assert(size(Y0, 1) == mm, 'Y0 must have mm = %d rows (grid points).', mm);

% Initial coefficients via least-squares
theta = XX_poly \ Y0;

POL.theta = theta;
POL.algo = 'cheb_tmi';
POL.dd = dd;

damp = options.Damping;
exitflag = 0;

for iter = 1:options.MaxIter
    % Evaluate current policy on grid
    Y_old = XX_poly * theta;

    % Evaluate residuals
    R = fun_res(Y_old, xx, POL, GRID);

    % New target: Y_new = Y_old - R (Newton-like step on residual)
    Y_new = Y_old - R;

    % Dampening
    if damp > 0
        Y_new = (1 - damp) * Y_new + damp * Y_old;
    end

    % Update coefficients
    theta_new = XX_poly \ Y_new;

    % Check convergence
    diff_val = max(abs(theta_new - theta), [], 'all');
    res_val  = max(abs(R), [], 'all');

    if options.Display && mod(iter, 50) == 0
        fprintf('  ProjSolve iter %4d: diff = %.2e, max|R| = %.2e\n', iter, diff_val, res_val);
    end

    theta = theta_new;
    POL.theta = theta;

    if diff_val < options.Tol && res_val < options.Tol
        exitflag = 1;
        if options.Display
            fprintf('  ProjSolve converged in %d iterations (diff=%.2e, res=%.2e).\n', ...
                iter, diff_val, res_val);
        end
        break
    end
end

if exitflag == 0 && options.Display
    fprintf('  ProjSolve: max iterations reached (diff=%.2e, res=%.2e).\n', diff_val, res_val);
end

end

% =========================================================================
% LOCAL: build complete polynomial multi-index
% =========================================================================
function poly_elem = build_poly_elements(nn, ord)
    % Generate all multi-indices (k1,...,kn) with sum <= ord
    if nn == 1
        poly_elem = (0:ord)';
        return
    end
    % Recursive construction
    elems = zeros(0, nn);
    idx = zeros(1, nn);
    elems = add_indices(elems, idx, 1, nn, ord);
    poly_elem = elems;
end

function elems = add_indices(elems, idx, dim, nn, ord)
    if dim > nn
        if sum(idx) <= ord
            elems = [elems; idx]; %#ok<AGROW>
        end
        return
    end
    for k = 0:ord
        idx(dim) = k;
        if sum(idx(1:dim)) > ord; break; end
        elems = add_indices(elems, idx, dim+1, nn, ord);
    end
end

% =========================================================================
% LOCAL: evaluate Chebyshev basis at points
% =========================================================================
function Phi = eval_cheb_basis(xx_dw, poly_elem)
    [mm, nn] = size(xx_dw);
    pp = size(poly_elem, 1);

    % Precompute univariate Chebyshev values T_k(x_d) for each dimension
    max_ord = max(poly_elem(:));
    T_vals = cell(1, nn);
    for d = 1:nn
        T_d = zeros(mm, max_ord + 1);
        T_d(:, 1) = 1;
        if max_ord >= 1
            T_d(:, 2) = xx_dw(:, d);
        end
        for k = 2:max_ord
            T_d(:, k+1) = 2 * xx_dw(:, d) .* T_d(:, k) - T_d(:, k-1);
        end
        T_vals{d} = T_d;
    end

    % Build basis: product of univariate Chebyshev polynomials
    Phi = ones(mm, pp);
    for j = 1:pp
        for d = 1:nn
            k = poly_elem(j, d);
            Phi(:, j) = Phi(:, j) .* T_vals{d}(:, k+1);
        end
    end
end
