function [fh, xh, gh, H, itct, fcount, retcodeh] = CsminWel(fcn, x0, H0, grad, crit, nit, varargin)
%CSMINWEL Chris Sims' quasi-Newton minimizer with BFGS Hessian update
%
%   Robust unconstrained minimizer designed for likelihood optimization
%   of DSGE / macroeconomic models.  Handles cliff edges, bad gradients,
%   and near-singular Hessians gracefully.
%
%   Reference:
%       Sims, C.A. (2000). "Using a Likelihood Perspective to Sharpen
%       Econometric Discourse." Journal of Econometrics.
%       Original code: www.princeton.edu/~sims
%
%   Usage:
%       [fhat,xhat,ghat,Hhat] = quantecon.optimization.CsminWel(@objfcn, x0, eye(n), [], 1e-8, 200);
%
%   Inputs:
%       fcn   - function handle  f = fcn(x, varargin{:})
%       x0    - (n x 1) initial parameter vector
%       H0    - (n x n) initial inverse Hessian (positive definite)
%       grad  - function handle for gradient, OR [] for numerical gradient
%       crit  - convergence criterion (improvement < crit => stop)
%       nit   - maximum number of iterations
%       varargin - extra arguments passed to fcn (and grad)
%
%   Outputs:
%       fh       - minimized function value
%       xh       - (n x 1) optimal parameter vector
%       gh       - (n x 1) gradient at optimum
%       H        - (n x n) inverse Hessian at optimum
%       itct     - iteration count
%       fcount   - function evaluation count
%       retcodeh - return code (0=normal, 1=zero grad, 2-7 = see source)

% [FIX]: All eval() calls replaced with feval / direct function handle calls
% [FIX]: All save() calls removed — no disk I/O during optimization

if ~isa(fcn, 'function_handle')
    error('quantecon:optimization:CsminWel:badFcn', ...
          'fcn must be a function handle, not a string.');
end

x0 = x0(:);
n = length(x0);
useNumGrad = isempty(grad);

itct   = 0;
fcount = 0;

% Initial function evaluation
f0 = fcn(x0, varargin{:});
fcount = fcount + 1;
if f0 > 1e50
    warning('quantecon:optimization:CsminWel:badInit', 'Bad initial parameter.');
    fh = f0; xh = x0; gh = NaN(n,1); H = H0; retcodeh = -1;
    return
end

% Initial gradient
if useNumGrad
    [g, badg] = numgrad_local(fcn, x0, varargin{:});
    fcount = fcount + n;
else
    [g, badg] = grad(x0, varargin{:});
end

x = x0;
f = f0;
H = H0;
done = false;

while ~done
    itct = itct + 1;

    % Line search
    [f1, x1, fc, retcode1] = csminit_local(fcn, x, f, g, badg, H, varargin{:});
    fcount = fcount + fc;

    g1 = []; g2 = []; g3 = [];
    badg1 = 1; badg2 = 1; badg3 = 1;
    f2 = f; f3 = f;
    x2 = x; x3 = x;
    retcode2 = 101; retcode3 = 101;
    wall1 = 1;

    if retcode1 ~= 1
        if retcode1 == 2 || retcode1 == 4
            wall1 = 1; badg1 = 1;
        else
            if useNumGrad
                [g1, badg1] = numgrad_local(fcn, x1, varargin{:});
                fcount = fcount + n;
            else
                [g1, badg1] = grad(x1, varargin{:});
            end
            wall1 = badg1;
        end

        if wall1
            % Cliff edge — perturb search direction
            Hcliff = H + diag(diag(H) .* rand(n, 1));
            [f2, x2, fc, retcode2] = csminit_local(fcn, x, f, g, badg, Hcliff, varargin{:});
            fcount = fcount + fc;

            if f2 < f
                if retcode2 == 2 || retcode2 == 4
                    wall2 = 1; badg2 = 1;
                else
                    if useNumGrad
                        [g2, badg2] = numgrad_local(fcn, x2, varargin{:});
                        fcount = fcount + n;
                    else
                        [g2, badg2] = grad(x2, varargin{:});
                    end
                    wall2 = badg2;
                end

                if wall2
                    % Try traversing
                    if norm(x2 - x1) < 1e-13
                        f3 = f; x3 = x; badg3 = 1; retcode3 = 101;
                    else
                        gcliff = ((f2 - f1) / (norm(x2 - x1)^2)) * (x2 - x1);
                        [f3, x3, fc, retcode3] = csminit_local(fcn, x, f, gcliff, 0, eye(n), varargin{:});
                        fcount = fcount + fc;
                        if retcode3 == 2 || retcode3 == 4
                            badg3 = 1;
                        else
                            if useNumGrad
                                [g3, badg3] = numgrad_local(fcn, x3, varargin{:});
                                fcount = fcount + n;
                            else
                                [g3, badg3] = grad(x3, varargin{:});
                            end
                        end
                    end
                else
                    f3 = f; x3 = x; badg3 = 1; retcode3 = 101;
                end
            else
                f3 = f; x3 = x; badg3 = 1; retcode3 = 101;
            end
        else
            f2 = f; f3 = f; badg2 = 1; badg3 = 1;
            retcode2 = 101; retcode3 = 101;
        end
    end

    % Pick best candidate
    if f3 < f && badg3 == 0
        fh = f3; xh = x3; gh = g3; badgh = badg3; retcodeh = retcode3;
    elseif f2 < f && badg2 == 0
        fh = f2; xh = x2; gh = g2; badgh = badg2; retcodeh = retcode2;
    elseif f1 < f && badg1 == 0
        fh = f1; xh = x1; gh = g1; badgh = badg1; retcodeh = retcode1;
    else
        [fh, ih] = min([f1, f2, f3]);
        candidates_x = {x1, x2, x3};
        candidates_g = {g1, g2, g3};
        retcodes     = [retcode1, retcode2, retcode3];
        xh = candidates_x{ih};
        retcodeh = retcodes(ih);
        gh_cand = candidates_g{ih};
        if isempty(gh_cand)
            if useNumGrad
                [gh, ~] = numgrad_local(fcn, xh, varargin{:});
                fcount = fcount + n;
            else
                [gh, ~] = grad(xh, varargin{:});
            end
        else
            gh = gh_cand;
        end
        badgh = 1;
    end

    % BFGS update
    stuck = (abs(fh - f) < crit);
    if ~badg && ~badgh && ~stuck
        H = bfgsi_local(H, gh - g, xh - x);
    end

    % Termination checks
    if itct > nit
        done = true;
    elseif stuck
        done = true;
    end

    f = fh;
    x = xh;
    g = gh;
    badg = badgh;
end
end


%% ========== LOCAL FUNCTION: csminit ==========
function [fhat, xhat, fcount, retcode] = csminit_local(fcn, x0, f0, g0, badg, H0, varargin)
%CSMINIT_LOCAL Line-search initialization for CsminWel

ANGLE   = 0.005;
THETA   = 0.3;
FCHANGE = 1000;
MINLAMB = 1e-9;
MINDFAC = 0.01;

fcount = 0;
xhat   = x0;
fhat   = f0;
gnorm  = norm(g0);

if (gnorm < 1e-12) && ~badg
    retcode = 1;
    return
end

dx = -H0 * g0;
dxnorm = norm(dx);
if dxnorm > 1e12
    dx = dx * FCHANGE / dxnorm;
end

dfhat = dx' * g0;

if ~badg
    a = -dfhat / (gnorm * dxnorm);
    if a < ANGLE
        dx = dx - (ANGLE * dxnorm / gnorm + dfhat / (gnorm^2)) * g0;
        dfhat  = dx' * g0;
        dxnorm = norm(dx);
    end
end

lambda     = 1;
done       = false;
factor     = 3;
shrink     = true;
lambdaMin  = 0;
lambdaMax  = inf;
lambdaPeak = 0;
fPeak      = f0;

while ~done
    dxtest = x0 + dx * lambda;
    f = fcn(dxtest, varargin{:});
    fcount = fcount + 1;

    if f < fhat
        fhat = f;
        xhat = dxtest;
    end

    shrinkSignal = (~badg && (f0 - f < max([-THETA * dfhat * lambda, 0]))) ...
                 || (badg && (f0 - f) < 0);
    growSignal   = ~badg && (lambda > 0) && (f0 - f > -(1 - THETA) * dfhat * lambda);

    if shrinkSignal && ((lambda > lambdaPeak) || (lambda < 0))
        if (lambda > 0) && (~shrink || (lambda / factor <= lambdaPeak))
            shrink = true;
            factor = factor^0.6;
            while lambda / factor <= lambdaPeak
                factor = factor^0.6;
            end
            if abs(factor - 1) < MINDFAC
                retcode = 2; done = true;
            end
        end
        if (lambda < lambdaMax) && (lambda > lambdaPeak)
            lambdaMax = lambda;
        end
        lambda = lambda / factor;
        if abs(lambda) < MINLAMB
            if (lambda > 0) && (f0 <= fhat)
                lambda = -lambda * factor^6;
            else
                retcode = ternary(lambda < 0, 6, 3);
                done = true;
            end
        end

    elseif (growSignal && lambda > 0) || (shrinkSignal && (lambda <= lambdaPeak) && (lambda > 0))
        if shrink
            shrink = false;
            factor = factor^0.6;
            if abs(factor - 1) < MINDFAC
                retcode = 4; done = true;
            end
        end
        if (f < fPeak) && (lambda > 0)
            fPeak = f;
            lambdaPeak = lambda;
            if lambdaMax <= lambdaPeak
                lambdaMax = lambdaPeak * factor^2;
            end
        end
        lambda = lambda * factor;
        if abs(lambda) > 1e20
            retcode = 5; done = true;
        end
    else
        done = true;
        retcode = ternary(factor < 1.2, 7, 0);
    end
end
end


%% ========== LOCAL FUNCTION: bfgsi ==========
function H = bfgsi_local(H0, dg, dx)
%BFGSI_LOCAL Inverse-Hessian BFGS update (Sims, 1996)

dg = dg(:);
dx = dx(:);
Hdg  = H0 * dg;
dgdx = dg' * dx;

if abs(dgdx) > 1e-12
    H = H0 + (1 + (dg' * Hdg) / dgdx) * (dx * dx') / dgdx ...
        - (dx * Hdg' + Hdg * dx') / dgdx;
else
    H = H0;
end
% [FIX]: Removed save H.dat from original bfgsi.m
end


%% ========== LOCAL FUNCTION: numgrad ==========
function [g, badg] = numgrad_local(fcn, x, varargin)
%NUMGRAD_LOCAL Forward-difference numerical gradient
% [FIX]: All eval() replaced with direct function-handle calls

delta = 1e-6;
n = length(x);
g = zeros(n, 1);
f0 = fcn(x, varargin{:});
badg = false;

for i = 1:n
    ei    = zeros(n, 1);
    ei(i) = delta;
    g0 = (fcn(x + ei, varargin{:}) - f0) / delta;
    if abs(g0) < 1e15
        g(i) = g0;
    else
        g(i) = 0;
        badg = true;
    end
end
end


%% ========== LOCAL FUNCTION: ternary ==========
function v = ternary(cond, a, b)
%TERNARY Inline conditional
if cond
    v = a;
else
    v = b;
end
end
