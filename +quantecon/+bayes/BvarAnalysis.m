classdef BvarAnalysis
    %BVARANALYSIS Utility for Post-Estimation BVAR features
    %   Covers IRF, FEVD, and Historical Decomposition.

    methods (Static)
        function irfs = irf(coeffs, Sigma, p, horizon, ident)
            %IRF Impulse Response Functions
            %   ident: "chol" (Cholesky) or "none" (Reduced form)

            arguments
                coeffs (:,:) double % [K x N] where K = 1 + N*p
                Sigma (:,:) double % [N x N]
                p (1,1) double
                horizon (1,1) double = 20
                ident string = "chol"
            end

            N = size(Sigma, 1);

            % Companion Form
            [F, ~] = quantecon.bayes.BvarAnalysis.get_companion(coeffs, N, p);

            % Identification Matrix
            if strcmpi(ident, "chol")
                P = chol(Sigma, 'lower');
            else
                P = eye(N);
            end

            irfs = zeros(N, N, horizon);
            if p > 1
                J = [eye(N), zeros(N, N*(p-1))];
            else
                J = eye(N);
            end

            for h = 1:horizon
                irfs(:, :, h) = J * (F^(h-1)) * J' * P;
            end
        end

        function fevd = fevd(coeffs, Sigma, p, horizon)
            %FEVD Forecast Error Variance Decomposition
            irfs = quantecon.bayes.BvarAnalysis.irf(coeffs, Sigma, p, horizon, "chol");
            N = size(Sigma, 1);

            fevd = zeros(N, N, horizon);
            mse = zeros(N, horizon);

            for h = 1:horizon
                % Contribution of shock j to variance of variable i at horizon h
                for j = 1:N
                    for i = 1:N
                        step_val = sum(reshape(irfs(i, j, 1:h), [], 1).^2);
                        fevd(i, j, h) = step_val;
                    end
                end
                mse(:, h) = sum(fevd(:, :, h), 2);
                fevd(:, :, h) = fevd(:, :, h) ./ mse(:, h);
            end
        end

        function hd = hdecomp(Y, X, coeffs, N, p)
            %HDECOMP Historical Decomposition
            %   Y: T x N (actual data)
            %   X: T x K (regressors)

            T = size(Y, 1);
            res = Y - X * coeffs;

            % HD = Contribution of each shock to each variable over time
            hd = zeros(T, N, N); % [Time x Variable x ShockSource]

            % Pre-calculate VMA matrices Psi
            [F, ~] = quantecon.bayes.BvarAnalysis.get_companion(coeffs, N, p);
            if p > 1
                J = [eye(N), zeros(N, N*(p-1))];
            else
                J = eye(N);
            end

            % Identification
            Sigma = (res' * res) / T;
            P = chol(Sigma, 'lower');
            shocks = (P \ res')'; % T x N structural shocks

            for t = 1:T
                for s = 0:t-1
                    Psi = J * (F^s) * J';
                    for j = 1:N
                        contribution = Psi * P(:, j) * shocks(t-s, j);
                        hd(t, :, j) = hd(t, :, j) + contribution';
                    end
                end
            end
        end

        function forecasts = cforecast(coeffs, Sigma, p, horizon, cond_idx, cond_paths)
            %CFORECAST Conditional Forecasting (Waggoner & Zha 1999)
            %   cond_idx: indices of variables with conditions
            %   cond_paths: [horizon x length(cond_idx)] matrix of paths

            arguments
                coeffs (:,:) double
                Sigma (:,:) double
                p (1,1) double
                horizon (1,1) double
                cond_idx (1,:) double
                cond_paths (:,:) double
            end

            N = size(Sigma, 1);

            % Initialize with zero starting values for state
            forecasts = zeros(horizon, N);
            X_curr = zeros(1, N*p);

            for h = 1:horizon
                Y_next = [1, X_curr] * coeffs;
                forecasts(h, :) = Y_next;
                % Override conditions
                forecasts(h, cond_idx) = cond_paths(h, :);
                % Update state
                if p > 1
                    X_curr = [forecasts(h, :), X_curr(1:end-N)];
                else
                    X_curr = forecasts(h, :);
                end
            end
        end
    end

    methods (Static, Access = private)
        function [F, K] = get_companion(coeffs, N, p)
            % Companion matrix F of size (N*p x N*p)
            % coeffs is [K x N] where K = 1+N*p (with constant) or N*p (no constant)
            K_size = size(coeffs, 1);
            if K_size == N*p + 1
                A = coeffs(2:end, :)'; % [N x N*p]
                K = coeffs(1, :)'; % Constants
            else
                A = coeffs'; % [N x N*p]
                K = zeros(N, 1);
            end

            if p > 1
                F = [A; eye(N*(p-1)), zeros(N*(p-1), N)];
            else
                F = A;
            end
        end
    end
end
