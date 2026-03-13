classdef Covar < handle
    %COVAR Conditional Value at Risk (CoVaR)
    %
    %   Implements CoVaR using Quantile Regression (Adrian & Brunnermeier 2016).
    %
    %   Model:
    %       X_q^{sys|i} = \alpha_q + \beta_q * X^i
    %       \Delta CoVaR_q^i = CoVaR_q^{sys|X^i=VaR_q^i} - CoVaR_q^{sys|X^i=Median^i}
    %                        = \beta_q * (VaR_q^i - Median^i)
    %
    %   Usage:
    %       mdl = quantecon.risk.Covar(0.05);
    %       % R_sys: System returns, R_inst: Institution returns
    %       results = mdl.estimate(R_sys, R_inst);

    properties
        Quantile (1,1) double {mustBeGreaterThan(Quantile, 0), mustBeLessThan(Quantile, 1)} = 0.05
        Results struct
    end

    methods
        function obj = Covar(q)
            if nargin > 0
                obj.Quantile = q;
            end
        end

        function results = estimate(obj, R_sys, R_inst, StateVars)
            %ESTIMATE Estimate CoVaR and Delta CoVaR
            %
            %   Input:
            %       R_sys: (Tx1) System returns
            %       R_inst: (Tx1) Institution returns
            %       StateVars: (TxK) [Optional] Macro state variables

            arguments
                obj
                R_sys (:,1) double
                R_inst (:,1) double
                StateVars (:,:) double = []
            end

            T = length(R_sys);
            assert(length(R_inst) == T, 'Returns must have same length');

            q = obj.Quantile;

            % 1. Estimate VaR_q^i (Institution VaR)
            % If StateVars present: R_inst = a + b*StateVars
            if isempty(StateVars)
                % Constant VaR (Unconditional)
                var_inst = quantile(R_inst, q);
                median_inst = median(R_inst);
                VAR_i_q = repmat(var_inst, T, 1);
                Median_i = repmat(median_inst, T, 1);

                X = [ones(T,1), R_inst];
            else
                % Conditional VaR using Quantile Regression
                X_inst = [ones(T,1), StateVars(1:end,:)]; % Lagged? Usually contemporaneous state vars in Adrian/Brunnermeier?
                % Paper: X_t^i = a + b*M_{t-1} + e. usually lagged state vars.

                beta_var = obj.quantile_regression(R_inst, X_inst, q);
                VAR_i_q = X_inst * beta_var;

                beta_med = obj.quantile_regression(R_inst, X_inst, 0.5);
                Median_i = X_inst * beta_med;

                X = [ones(T,1), R_inst, StateVars];
            end

            % 2. Estimate CoVaR System: R_sys = a + b*R_inst + c*StateVars
            beta_sys = obj.quantile_regression(R_sys, X, q);

            % 3. Calculate CoVaR and Delta CoVaR

            % CoVaR_q^{sys|X^i=VaR_q^i}
            % X_cond = [1, VAR_i_q, StateVars]
            if isempty(StateVars)
                CoVaR = beta_sys(1) + beta_sys(2) * VAR_i_q;
                CoVaR_Med = beta_sys(1) + beta_sys(2) * Median_i;
            else
                % beta_sys: [const, R_inst, StateVars...]
                % Contribution from StateVars
                StateContrib = X(:, 3:end) * beta_sys(3:end);

                CoVaR = beta_sys(1) + beta_sys(2) * VAR_i_q + StateContrib;
                CoVaR_Med = beta_sys(1) + beta_sys(2) * Median_i + StateContrib;
            end

            DeltaCoVaR = CoVaR - CoVaR_Med;

            results.VaR_inst = VAR_i_q;
            results.CoVaR = CoVaR;
            results.DeltaCoVaR = DeltaCoVaR;
            results.Beta_System = beta_sys;

            obj.Results = results;
        end
    end

    methods (Access = private)
        function b = quantile_regression(~, y, X, q)
            % Simple Iterative Reweighted Least Squares for Quantile Regression
            % Min sum |y - Xb| * (q if resid>0 else 1-q)

            [n, k] = size(X);
            b = (X' * X) \ (X' * y); % OLS Start

            max_iter = 100;
            tol = 1e-6;

            for iter = 1:max_iter
                b_old = b;
                resid = y - X*b;

                % Weighting
                % w = q/|e| if e>0, (1-q)/|e| if e<0
                % Approx |e| with sqrt(e^2 + eps) for differentiability

                resid(abs(resid) < 1e-6) = 1e-6; % Avoid div by zero
                weights = zeros(n,1);

                pos_idx = resid > 0;
                weights(pos_idx) = q ./ abs(resid(pos_idx));
                weights(~pos_idx) = (1-q) ./ abs(resid(~pos_idx));

                % Weighted Least Squares
                % b = (X'WX)^-1 X'Wy
                % Let W = diag(weights)

                % Efficient WLS: X_w = sqrt(w) .* X, y_w = sqrt(w) .* y
                sw = sqrt(weights);
                X_w = sw .* X;
                y_w = sw .* y;

                b = (X_w' * X_w) \ (X_w' * y_w);

                if max(abs(b - b_old)) < tol
                    break;
                end
            end
        end
    end
end
