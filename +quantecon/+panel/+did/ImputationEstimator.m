classdef ImputationEstimator
    % IMPUTATIONESTIMATOR Borusyak, Jaravel, Spiess (2024) Estimator
    %
    %   Estimates ATT by imputing counterfactual outcomes for treated units
    %   using a model fitted on untreated observations (D=0).
    %
    %   Usage:
    %       mdl = quantecon.panel.did.ImputationEstimator();
    %       res = mdl.estimate(y, treat, time, id, X);

    methods
        function res = estimate(obj, y, treat, time, id, X)
            arguments
                obj
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
                X (:,:) double = []
            end

            % 1. Identify Estimation Sample (D=0)
            % Untreated observations: Not currently treated.
            % Note: In BJS, "untreated" means D_it = 0.
            % This includes Never-Treated and Not-Yet-Treated.

            mask_0 = (treat == 0);
            mask_1 = (treat == 1);

            if sum(mask_0) == 0
                error('No untreated observations found.');
            end

            y_0 = y(mask_0);
            time_0 = time(mask_0);
            id_0 = id(mask_0);

            % 2. Fit Model on D=0
            % y_it(0) = alpha_i + lambda_t + X_it * beta + epsilon
            % Utilize FixedEffects class for efficient estimation

            fe_mdl = quantecon.panel.FixedEffects();
            % Need to pass X if it exists
            if isempty(X)
                % Just Fixed Effects
                % FE model expects some X? estimate(y, X, ...)
                % If X is empty, FixedEffects might need a dummy or handle it.
                % Let's create a dummy Regressor of zeros?
                % Or use a specialized FE solver for just alpha+lambda.
                % quantecon.panel.FixedEffects currently requires X.
                % Let's construct dummy X = ones? No, that's collinear with intercept.
                % Let's use random noise with near-zero weight? No.

                % Fallback: Create a column of zeros and ignore coefficient?
                X_0 = zeros(length(y_0), 1);
                res_0 = fe_mdl.estimate(y_0, X_0, id_0, time_0);
                % Alpha and Lambda are in the residuals?
                % Use predictions: y_hat = alpha_i + lambda_t.
                % FixedEffects returns y_hat? Check.
                beta_hat = 0; % effectively
            else
                X_0 = X(mask_0, :);
                res_0 = fe_mdl.estimate(y_0, X_0, id_0, time_0);
                beta_hat = res_0.Results.Coefficients;
            end

            % 3. Impute Y(0) for Treated (D=1)
            % y_it(0)_hat = alpha_i_hat + lambda_t_hat + X_it * beta_hat
            % We need to extract FEs from the trained model.
            % FixedEffects might not expose FEs directly if it uses Within Transformation.
            % If it uses Within, we have: predict(new_data) is hard without FEs.

            % ALTERNATIVE: Explicitly solve for FEs or use lscov with dummies.
            % Or, predict y_hat using the structure:
            % y_hat = y_bar_i + y_bar_t - y_bar_bar + (X - X_bar)*beta
            % This is valid for balanced panels. Unbalanced is harder.

            % Implementation decision:
            % Use a simpler approach for Imputation: High-Dimensional OLS with simple dummies?
            % Or iterate?
            % Let's assume FixedEffects class has a 'predict' or 'getFE' method.
            % If not, we iterate:
            % y - X*b = alpha + lambda + e
            % Estimate alpha, lambda from residuals.

            % Get residuals from estimation
            u_0 = res_0.Results.Residuals;
            % u_0 contains (alpha_i + lambda_t + e_it) if we only removed X*beta?
            % No, Within estimator removes FEs. u_0 approximates e_it.
            % Wait, we need y_hat = alpha + lambda + X*beta.
            % y_0 = y_hat + e.
            % So y_hat_0 = y_0 - res_0.Residuals.

            % We need to extrapolate y_hat to D=1 sample.
            % This implies we need alpha_i for treated units and lambda_t for post periods.
            % - Treated units have pre-periods (in D=0). So alpha_i is identifiable.
            % - Post periods have control units (in D=0). So lambda_t is identifiable.
            % This is the "Imputation" logic.

            % Recover FE Method:
            % 1. Compute offsets:
            %    resid_XB = y - X * beta_hat; (All units)
            %    Use D=0 to estimate FEs on resid_XB.
            % 2. Algorithm:
            %    Iterate or use connectedness.
            %    Simple case: alpha_i = mean(resid_XB_i) - mean(lambda) ??
            %    Jointly solve: resid_XB ~ TimeDummies + UnitDummies

            % For speed in MATLAB, use lscov with sparse matrix?
            % Or simpler:
            % Or simpler:
            if isempty(X)
                Y_star = y;
            else
                Y_star = y - (X * beta_hat);
            end

            % Solve Y_star ~ FEs on D=0
            % Construct sparse design matrix for FEs
            [ids, ~, idIdx] = unique(id);
            [times, ~, timeIdx] = unique(time);

            % Subset to D=0
            id_0 = idIdx(mask_0);
            t_0 = timeIdx(mask_0);
            y_star_0 = Y_star(mask_0);

            % Design: [D_id, D_time]
            % Drop first of one to identify
            D_id = sparse(1:length(id_0), id_0, 1, length(id_0), length(ids));
            D_t = sparse(1:length(t_0), t_0, 1, length(t_0), length(times));

            % Solve [D_id, D_t] * [alpha; lambda] = y_star_0
            % Rank deficiency handling needed.
            Design = [D_id, D_t];
            params = Design \ y_star_0;

            alpha_hat = params(1:length(ids));
            lambda_hat = params(length(ids)+1:end);

            % 4. Predict Y(0) for D=1
            id_1 = idIdx(mask_1);
            t_1 = timeIdx(mask_1);

            y0_hat = alpha_hat(id_1) + lambda_hat(t_1);
            if ~isempty(X)
                y0_hat = y0_hat + X(mask_1, :) * beta_hat;
            end

            % 5. ATT = Mean(y(1) - y0_hat)
            y_1 = y(mask_1);
            tau_it = y_1 - y0_hat;
            att = mean(tau_it);

            res = struct('ATT', att, 'Tau_it', tau_it, 'Beta', beta_hat);
        end
    end
end
