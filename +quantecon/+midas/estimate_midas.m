function results = estimate_midas(y, X_hf, V_lf, opts)
%ESTIMATE_MIDAS Core estimation logic for MIDAS-NLS
%
%   Mathematical Foundation:
%   Minimizes ||y - Z * beta * w(theta) - V * rho||^2
%   where Z is HF data, w(theta) are weights, V is LF data.

% Core estimation engine for MIDAS-NLS
k_lf = size(V_lf, 2);

% Polynomial Specific setup
switch opts.Polynomial
    case "Beta"
        % theta = [theta1; theta2; theta3]
        % For BetaCase 3: theta1=1, theta3=0 fixed. Only theta2 optimized.
        % Starting values from MIDAS analytic 0.1
        % theta1, theta2, theta3, beta, rho
        init = [1; 5; 0; 1; zeros(k_lf, 1)];
        lb = [0.1; 1.2; -3; -Inf; -Inf(k_lf, 1)];
        ub = [2.0; 20.0; 3; Inf; Inf(k_lf, 1)];

        % Overwrite fixed params based on BetaCase
        if opts.BetaCase == 1 || opts.BetaCase == 3
            init(1) = 1; lb(1) = 1; ub(1) = 1;
        end
        if opts.BetaCase == 2 || opts.BetaCase == 3
            init(3) = 0; lb(3) = 0; ub(3) = 0;
        end

        obj_func = @(params) quantecon.midas.derivatives(params, y, X_hf, V_lf, "Beta", opts.BetaCase);
        optim_opts = optimoptions("fmincon", "GradObj", "on", "Algorithm", "interior-point", ...
            "Hessian", "user-supplied", "HessianFcn", @(p, lam) hess_adapter(p, y, X_hf, V_lf, "Beta", opts.BetaCase), ...
            "Display", "off");

    case "ExpAlmon"
        % theta = [theta1; theta2]
        init = [0.25; -0.25; 1; zeros(k_lf, 1)];
        lb = [0; -1; -5; -Inf(k_lf, 1)];
        ub = [1; 0; 5; Inf(k_lf, 1)];

        obj_func = @(params) quantecon.midas.derivatives(params, y, X_hf, V_lf, "ExpAlmon");
        optim_opts = optimoptions("fminunc", "GradObj", "on", "Algorithm", "trust-region", ...
            "Hessian", "user-supplied", "HessianFcn", "objective", "Display", "off");

    case "Almon"
        % Standard Almon is linear, can use OLS after transformation
        results = estimate_almon_ols(y, X_hf, V_lf, 3);
        return;
end

% Optimization
if opts.MultiStart
    problem = createOptimProblem("fmincon", "objective", obj_func, "x0", init, "lb", lb, "ub", ub, "options", optim_opts);
    ms = MultiStart("Display", "off");
    [params_hat, fval] = run(ms, problem, opts.Runs);
else
    if opts.Polynomial == "ExpAlmon"
        [params_hat, fval] = fminunc(obj_func, init, optim_opts);
    else
        [params_hat, fval] = fmincon(obj_func, init, [], [], [], [], lb, ub, [], optim_opts);
    end
end

% Store Results
results.Parameters = params_hat;
results.SSR = fval;
[~, ~, ~, weights] = quantecon.midas.derivatives(params_hat, y, X_hf, V_lf, opts.Polynomial, opts.BetaCase);
results.Weights = weights;

% Calculate yhat
if opts.Polynomial == "ExpAlmon"
    beta_val = params_hat(3);
    rho_val = params_hat(4:end);
else
    beta_val = params_hat(4);
    rho_val = params_hat(5:end);
end
results.YHat = X_hf * (beta_val * weights) + V_lf * rho_val;
results.Residuals = y - results.YHat;
results.R2 = 1 - results.SSR / sum((y - mean(y)).^2);

end

function H = hess_adapter(params, y, X_hf, V_lf, poly, beta_case)
[~, ~, H] = quantecon.midas.derivatives(params, y, X_hf, V_lf, poly, beta_case);
end

function results = estimate_almon_ols(y, X, V, degree)
% Simple OLS based Almon for compatibility
T = size(y, 1);
p = size(X, 2);
Z = zeros(T, degree + 1);
k_grid = (1:p)';
for d = 0:degree
    Z(:, d+1) = X * (k_grid.^d);
end
all_X = [V, Z];
mdl = quantecon.base.Ols(y, all_X, "HasConstant", false);
results.Coefficients = mdl.Coefficients;
results.YHat = mdl.YHat;
results.Residuals = mdl.Residuals;
results.R2 = mdl.R2;
% Recover weights
a = mdl.Coefficients(size(V,2)+1:end);
Q = zeros(p, degree+1);
for d = 0:degree
    Q(:, d+1) = k_grid.^d;
end
results.Weights = Q * a;
end
