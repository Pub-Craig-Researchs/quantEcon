classdef PlsEstimator
    % PLSESTIMATOR Partial Least Squares Structural Equation Modeling
    %
    %   Implements the PLS-PM algorithm (Wold, 1982).
    %
    %   Usage:
    %       mdl = quantecon.sem.PlsEstimator(DP, DB);
    %       mdl = mdl.estimate(MV);

    properties
        DP (:,:) double % Outer Design Matrix (p x J) - Boolean
        DB (:,:) double % Path Design Matrix (J x J) - Boolean
        Mode (:,1) string = "A" % Measurement Mode ("A" or "B")
        Scheme (1,1) string = "path" % Weighting Scheme ("path", "centroid", "factor")
        MaxIter (1,1) double = 300
        Tolerance (1,1) double = 1e-6
        Results struct
    end

    methods
        function obj = PlsEstimator(dp, db)
            obj.DP = dp;
            obj.DB = db;
            obj.Results = struct();
        end

        function obj = estimate(obj, MV)
            % ESTIMATE Estimate PLS-SEM model
            %
            %   Inputs:
            %       MV - (n x p) Manifest Variables (Raw Data)

            arguments
                obj
                MV (:,:) double
            end

            % Standardization
            X = zscore(MV);
            [n, p] = size(X);
            J = size(obj.DP, 2);

            % Initialize Weights (Outer weights)
            W = obj.DP; % Start with design matrix

            % Symmetric Path Design Matrix
            DB_sym = obj.DB + triu(obj.DB)';

            iter = 0;
            diff = 1;

            while diff > obj.Tolerance && iter < obj.MaxIter
                iter = iter + 1;
                W_old = W;

                % 1. Latent Scores Approximation
                Y = X * W;
                Y = zscore(Y); % Normalize variance to 1

                % 2. Inner Weights (Structuring)
                switch obj.Scheme
                    case "centroid"
                        V = sign(DB_sym .* cov(Y));
                    case "factor"
                        V = DB_sym .* corr(Y);
                    case "path"
                        % Simplified path scheme (Wold)
                        % Real implementation needs regression for specific paths
                        % Fallback to correlation for now for robustness
                        V = DB_sym .* corr(Y);
                    otherwise
                        V = DB_sym .* corr(Y);
                end

                Z = Y * V; % Inner estimation

                % 3. Outer Weights Update
                for j = 1:J
                    idx = find(obj.DP(:, j));
                    if isempty(idx), continue; end

                    x_block = X(:, idx);
                    z_j = Z(:, j);

                    if obj.Mode == "A"
                        % Mode A: Reflective (Correlation)
                        w_new = (1/n) * (z_j' * x_block)';
                    else
                        % Mode B: Formative (Regression)
                        w_new = (x_block' * x_block) \ (x_block' * z_j);
                    end
                    W(idx, j) = w_new;
                end

                % Check convergence
                diff = sum(sum((W - W_old).^2));
            end

            % Finalize
            % Normalize weights
            for j = 1:J
                idx = find(obj.DP(:, j));
                if ~isempty(idx)
                    w_norm = norm(W(idx, j));
                    if w_norm > 0
                        W(idx, j) = W(idx, j) / w_norm;
                    end
                end
            end

            % Final Scores
            LV = X * W;

            % Path Coefficients (Inner Model)
            Beta = zeros(J, J);
            R2 = zeros(J, 1);

            for j = 1:J
                predictors = find(obj.DB(:, j)); % Incoming paths
                if ~isempty(predictors)
                    y_target = LV(:, j);
                    X_preds = LV(:, predictors);
                    b = (X_preds' * X_preds) \ (X_preds' * y_target);
                    Beta(predictors, j) = b;

                    % Simple R2
                    y_hat = X_preds * b;
                    R2(j) = 1 - sum((y_target - y_hat).^2) / sum((y_target - mean(y_target)).^2);
                end
            end

            obj.Results.Weights = W;
            obj.Results.Scores = LV;
            obj.Results.PathCoefs = Beta;
            obj.Results.R2 = R2;
            obj.Results.Iterations = iter;
        end
    end
end
