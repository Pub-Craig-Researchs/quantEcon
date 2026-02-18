classdef FixedEffects
    % FIXEDEFFECTS Panel Data Fixed Effects Estimator
    %
    %   Usage:
    %       mdl = quantecon.panel.FixedEffects('TimeEffects', true, 'Cluster', 'TwoWay');
    %       mdl = mdl.estimate(y, X, id, time);
    %
    %   Methodology:
    %       - One-Way FE: Standard Within Transformation.
    %       - Two-Way FE: Hybrid approach (Within Transformation for IDs + Dummy Variables for Time).
    %       - Inference: Robust, Clustered (Entity), or Two-Way Clustered Standard Errors.

    properties
        Method (1,1) string {mustBeMember(Method, ["Within", "FirstDifference"])} = "Within"
        TimeEffects (1,1) logical = false
        Cluster (1,1) string {mustBeMember(Cluster, ["Robust", "Entity", "TwoWay"])} = "Robust"
        Results struct
    end

    methods
        function obj = FixedEffects(varargin)
            p = inputParser;
            addParameter(p, "Method", "Within");
            addParameter(p, "TimeEffects", false);
            addParameter(p, "Cluster", "Robust");
            parse(p, varargin{:});

            obj.Method = p.Results.Method;
            obj.TimeEffects = p.Results.TimeEffects;
            obj.Cluster = p.Results.Cluster;
            obj.Results = struct();
        end

        function obj = estimate(obj, y, X, id, time)
            % ESTIMATE Estimate Fixed Effects Model
            %
            %   Inputs:
            %       y - (N*T x 1) Dependent variable
            %       X - (N*T x K) Independent variables
            %       id - (N*T x 1) Individual ID
            %       time - (N*T x 1) Time ID

            arguments
                obj
                y (:,1) double
                X (:,:) double
                id (:,1) double
                time (:,1) double = []
            end

            if obj.TimeEffects && isempty(time)
                error("Time variable must be provided for TimeEffects.");
            end

            % 1. Handle Time Effects (Hybrid Approach)
            % Add Time Dummies to X BEFORE de-meaning
            % To avoid collinearity, drop one time period (e.g., the first one found)
            if obj.TimeEffects
                [uTime, ~, tIdx] = unique(time);
                T_periods = length(uTime);
                if T_periods > 1
                    % Create dummies for 2..T (drop first)
                    % Size: N*T x (T-1)
                    TimeDummies = zeros(length(y), T_periods - 1);
                    for t = 2:T_periods
                        TimeDummies(:, t-1) = (tIdx == t);
                    end
                    X = [X, TimeDummies];
                end
            end

            % Data Transformation
            % Sort not strictly required for accumarray but good for structure
            [ids, ~, idx] = unique(id);
            N_entities = length(ids);
            T_vals = accumarray(idx, 1);

            % Calculate means
            y_mean = accumarray(idx, y) ./ T_vals;
            X_mean = zeros(N_entities, size(X, 2));
            for k = 1:size(X, 2)
                X_mean(:, k) = accumarray(idx, X(:, k)) ./ T_vals;
            end

            % Subtract means
            y_dem = y - y_mean(idx);
            X_dem = X - X_mean(idx, :);

            % OLS on de-meaned data
            % Formula: beta = (X_dem' * X_dem)^-1 * (X_dem' * y_dem)
            beta = (X_dem' * X_dem) \ (X_dem' * y_dem);

            % Store Coefficients (Separate Core X from Time Dummies)
            K_orig = size(X, 2) - (obj.TimeEffects * (length(unique(time)) - 1));
            obj.Results.Coefficients = beta(1:K_orig);
            if obj.TimeEffects
                obj.Results.TimeEffects = beta(K_orig+1:end);
            end

            % Residuals (Original Level)
            % e_it = (y_it - mean_y_i) - (x_it - mean_x_i)'beta
            % Note: This effectively calculates residuals of the FE model
            resid = y_dem - X_dem * beta;
            obj.Results.Residuals = resid;

            % R2 (Within)
            TSS = sum((y_dem - mean(y_dem)).^2);
            RSS = sum(resid.^2);
            obj.Results.R2_Within = 1 - RSS / TSS;

            % 2. Variance-Covariance Matrix Estimation
            % Cluster-Robust Variance Estimator
            % V = (X'X)^-1 * Omega * (X'X)^-1
            % Omega adjustment factor c = N / (N-1) * (M / (M-1)) ?
            % Standard: c = 1 for large N. But usually N/(N-K) * G/(G-1)


            % K_reg = length(beta) (includes X and Time Dummies if explicit)
            % K_total = Total parameters for DoF correction

            N_obs = length(y);
            K_reg = length(beta);

            K_absorbed = 0;
            if ~strcmp(obj.Method, "FirstDifference")
                K_absorbed = N_entities;
            end
            K_total = K_reg + K_absorbed;

            % Bread: XtX_inv
            % Note: XtX is K_reg x K_reg

            % Meat: Omega
            switch obj.Cluster
                case "Robust"
                    % Correction: N / (N - K_total)
                    c = N_obs / (N_obs - K_total);
                    Omega = (X_dem' .* (resid.^2)') * X_dem;
                    XtX = X_dem' * X_dem;
                    V = c * (XtX \ (Omega / XtX));

                case "Entity"
                    % Clustered by ID
                    Scores = X_dem .* resid; % N_obs x K_reg

                    % Sum by ID
                    G_scores = zeros(N_entities, K_reg);
                    for k = 1:K_reg
                        G_scores(:, k) = accumarray(idx, Scores(:, k));
                    end

                    Omega = G_scores' * G_scores;

                    % Correction: G / (G - 1) * (N - 1) / (N - K_total)
                    G = N_entities;
                    c = (G / (G - 1)) * ((N_obs - 1) / (N_obs - K_total));
                    XtX = X_dem' * X_dem;
                    V = c * (XtX \ (Omega / XtX));

                case "TwoWay"
                    % Two-Way Cluster (Entity + Time)

                    % 1. V_ent
                    Scores = X_dem .* resid;
                    G_scores_ent = zeros(N_entities, K_reg);
                    for k = 1:K_reg, G_scores_ent(:, k) = accumarray(idx, Scores(:, k)); end

                    Omega_ent = G_scores_ent' * G_scores_ent;
                    G_ent = N_entities;
                    c_ent = (G_ent / (G_ent - 1)) * ((N_obs - 1) / (N_obs - K_total));
                    XtX = X_dem' * X_dem;
                    V_ent = c_ent * (XtX \ (Omega_ent / XtX));

                    % 2. V_time
                    [~, ~, t_idx] = unique(time);
                    N_time = length(unique(t_idx));

                    G_scores_time = zeros(N_time, K_reg);
                    for k = 1:K_reg, G_scores_time(:, k) = accumarray(t_idx, Scores(:, k)); end

                    Omega_time = G_scores_time' * G_scores_time;
                    c_time = (N_time / (N_time - 1)) * ((N_obs - 1) / (N_obs - K_total));
                    XtX = X_dem' * X_dem;
                    V_time = c_time * (XtX \ (Omega_time / XtX));

                    % 3. V_intersect (Robust)
                    Omega_rob = (X_dem' .* (resid.^2)') * X_dem;
                    c_rob = N_obs / (N_obs - K_total);
                    XtX = X_dem' * X_dem;
                    V_int = c_rob * (XtX \ (Omega_rob / XtX));

                    V = V_ent + V_time - V_int;
            end

            % Store Coefficients (Separate Core X from Time Dummies)
            % K_original_vars = size(X_input, 2)
            % But we modified X argument.
            % We stored K_orig at line 94.
            % We need to access it. But it's local.
            % Reconstruct:
            K_orig = size(X, 2) - (obj.TimeEffects * (length(unique(time)) - 1));

            obj.Results.Covariance = V(1:K_orig, 1:K_orig);
            SE = sqrt(diag(obj.Results.Covariance));
            obj.Results.SE = SE;
            obj.Results.tStat = obj.Results.Coefficients ./ SE;
            obj.Results.pValue = 2 * (1 - tcdf(abs(obj.Results.tStat), N_obs - K_total));

        end
    end
end
