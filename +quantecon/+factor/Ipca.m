classdef Ipca
    % IPCA Instrumented Principal Component Analysis
    %
    %   Reference: Kelly, Pruitt, Su (2019) "Instrumented Principal Component Analysis"
    %
    %   Usage:
    %       mdl = quantecon.factor.Ipca(K);
    %       mdl = mdl.estimate(Returns, Characteristics);
    %       [Gamma, Factors] = mdl.infer();

    properties
        K (1,1) double {mustBeInteger, mustBePositive} = 1 % Number of factors
        Parameters struct
        Results struct
    end

    methods
        function obj = Ipca(k)
            obj.K = k;
            obj.Parameters = struct();
            obj.Results = struct();
        end

        function obj = estimate(obj, Y, X)
            % ESTIMATE Estimate IPCA model
            %
            %   Inputs:
            %       Y - (T x N) Excess returns
            %       X - (T x N x L) Characteristics (Instruments)

            arguments
                obj
                Y (:,:) double
                X double {mustBeNumeric} % 3D array check manual
            end

            [T, N] = size(Y);
            [Tx, Nx, L] = size(X);
            assert(T == Tx && N == Nx, 'Dimensions of Y and X must match');

            % Pre-process data
            % Centering not strictly required if factors absorb mean, but good practice

            % Initialize Gamma (PCA on characteristics)
            % stack X: (T*N) x L
            X_flat = reshape(permute(X, [2, 1, 3]), [], L);
            [U, ~, ~] = svds(X_flat, obj.K);
            Gamma_Old = U(1:L, 1:obj.K); % Initial guess

            % ALS Loop
            MaxIter = 100;
            Tol = 1e-4;

            for iter = 1:MaxIter
                % 1. Estimate Factors given Gamma
                % F_t = (Gamma' * Z_t' * Z_t * Gamma)^-1 * Gamma' * Z_t' * y_t
                Factors = zeros(T, obj.K);

                for t = 1:T
                    Xt = squeeze(X(t, :, :)); % N x L
                    yt = Y(t, :)';            % N x 1

                    Beta_t = Xt * Gamma_Old;  % N x K
                    Factors(t, :) = (Beta_t' * Beta_t) \ (Beta_t' * yt);
                end

                % 2. Estimate Gamma given Factors
                % LHS: sum(kron(x_t, f_t) * y_t)
                % RHS: sum(kron(x_t, f_t) * (kron(x_t, f_t))')
                Numer = zeros(L * obj.K, 1);
                Denom = zeros(L * obj.K);

                for t = 1:T
                    Xt = squeeze(X(t, :, :)); % N x L
                    ft = Factors(t, :)';      % K x 1
                    yt = Y(t, :)';            % N x 1

                    % Kronecker product equivalent for efficiency
                    % We need to solve for vec(Gamma)
                    % y_it = x_it * Gamma * f_t + e_it
                    % y_it = (f_t' \otimes x_it) * vec(Gamma)

                    Kron_XF = kron(ft', Xt); % N x (L*K)
                    Numer = Numer + Kron_XF' * yt;
                    Denom = Denom + Kron_XF' * Kron_XF;
                end

                Gamma_New_Vec = Denom \ Numer;
                Gamma_New = reshape(Gamma_New_Vec, L, obj.K);

                % 3. Enforce Orthonormality
                [U_g, ~, V_g] = svd(Gamma_New, 'econ');
                Gamma_New = U_g * V_g';

                % Check convergence
                diff = norm(Gamma_New - Gamma_Old, 'fro');
                if diff < Tol
                    break;
                end
                Gamma_Old = Gamma_New;
            end

            obj.Results.Gamma = Gamma_New;
            obj.Results.Factors = Factors;

            % Calculate R2
            Y_pred = zeros(T, N);
            for t = 1:T
                Xt = squeeze(X(t, :, :));
                ft = Factors(t, :)';
                Y_pred(t, :) = (Xt * Gamma_New * ft)';
            end
            TSS = sum(sum(Y.^2));
            RSS = sum(sum((Y - Y_pred).^2));
            obj.Results.R2 = 1 - RSS / TSS;
            obj.Results.R2 = 1 - RSS / TSS;
        end

        function [Gamma, Factors] = infer(obj)
            % INFER Return estimated factors and loadings
            if isempty(obj.Results) || ~isfield(obj.Results, 'Gamma')
                error('Model must be estimated first.');
            end
            Gamma = obj.Results.Gamma;
            Factors = obj.Results.Factors;
        end
    end
end
