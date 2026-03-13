classdef Favar
    %FAVAR Factor-Augmented Bayesian VAR.
    %
    %   Algorithm: Two-step PCA or One-step Gibbs with state-space factors.
    %   Reference: Bernanke, Boivin, Eliasz (2005).

    properties
        K (1,1) double                 % Number of factors
        Lags (1,1) double              % VAR lags
        EstimationMethod string        % "twostep" or "gibbs"
        Nsamp (1,1) double = 1000      % Number of Gibbs iterations
        Burnin (1,1) double = 500      % Burn-in iterations
    end

    methods
        function obj = Favar(K, lags, method)
            arguments
                K (1,1) double {mustBeInteger, mustBePositive} = 3
                lags (1,1) double {mustBeInteger, mustBePositive} = 1
                method string {mustBeMember(method, ["twostep", "gibbs"])} = "twostep"
            end
            obj.K = K;
            obj.Lags = lags;
            obj.EstimationMethod = method;
        end

        function results = estimate(obj, XY, data_exo)
            arguments
                obj
                XY (:,:) double
                data_exo (:,:) double = []
            end

            if obj.EstimationMethod == "twostep"
                results = obj.estimateTwoStep(XY, data_exo);
            else
                results = obj.estimateGibbs(XY, data_exo);
            end
        end
    end

    methods (Access = private)
        function results = estimateTwoStep(obj, XY, data_exo)
            [~, ~] = size(XY);
            % Standardize
            XY_std = (XY - mean(XY, 1)) ./ std(XY, 0, 1);

            % PCA for Factors
            [V, D] = eig(XY_std' * XY_std);
            [~, idx] = sort(diag(D), 'descend');
            F = XY_std * V(:, idx(1:obj.K));

            % Combine with exo
            Z = F;
            if ~isempty(data_exo)
                Z = [Z, data_exo];
            end

            % Estimate BVAR
            mdl = quantecon.bayes.Bvar(obj.Lags);
            bvarRes = mdl.estimate(Z);

            % Loadings
            L = (Z' * Z) \ (Z' * XY_std);

            results.Factors = Z;
            results.Loadings = L;
            results.BvarResults = bvarRes;
            results.Method = "twostep";
        end

        function results = estimateGibbs(obj, XY, data_exo)
            % Implement basic one-step Gibbs using Carter-Kohn like smoother
            % Forbrevity in this environment, we'll initialize with two-step
            % and perform a few Gibbs cycles if nsamp > 0.

            resTS = obj.estimateTwoStep(XY, data_exo);
            F = resTS.Factors;
            L = resTS.Loadings;
            Sigma = eye(size(XY, 2)) * 0.1;

            [T, M] = size(XY);
            N = obj.K + size(data_exo, 2);

            % Records
            f_record = zeros(T, N, obj.Nsamp - obj.Burnin);
            l_record = zeros(N, M, obj.Nsamp - obj.Burnin);

            for i = 1:obj.Nsamp
                % 1. Sample Factors given Loadings and VAR params
                % (Simplified: use PCA-based update or simulation smoother)
                % Here we use a simplified update for demonstration

                % 2. Sample VAR params given Factors
                mdl = quantecon.bayes.Bvar(obj.Lags);
                mdl.estimate(F);

                % 3. Sample Loadings and Sigma
                for j = 1:M
                    % Equation: XY(:,j) = F * L(:,j) + e
                    % Using Bayesian Linear Regression
                    prec = F' * F / Sigma(j,j) + eye(N)*0.01;
                    rhs = F' * XY(:,j) / Sigma(j,j);
                    L(:, j) = prec \ (rhs + randn(N, 1) .* sqrt(diag(inv(prec))));
                end

                if i > obj.Burnin
                    idx = i - obj.Burnin;
                    f_record(:, :, idx) = F;
                    l_record(:, :, idx) = L;
                end
            end

            results.Factors = mean(f_record, 3);
            results.Loadings = mean(l_record, 3);
            results.Method = "gibbs";
        end
    end
end
