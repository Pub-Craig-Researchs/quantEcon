classdef Spca < handle
    %SPCA Scaled PCA (Huang, Jiang, Li, Tong, Zhou 2022)
    %
    %   Implements Scaled PCA for dimension reduction.
    %   Logic:
    %   1. Regress target y on each predictor x_i to get slope beta_i.
    %   2. Scale x_i by beta_i: x_new_i = x_i * beta_i.
    %   3. Perform PCA on x_new.
    %
    %   Usage:
    %       mdl = quantecon.factor.Spca(3); % 3 factors
    %       res = mdl.estimate(Target, X);

    properties
        NumFactors (1,1) {mustBeInteger, mustBePositive} = 1
        Winsorize (1,1) logical = false
        LB (1,1) double = 1
        UB (1,1) double = 99
        Results struct
    end

    methods
        function obj = Spca(n, winsor)
            if nargin > 0
                obj.NumFactors = n;
            end
            if nargin > 1
                obj.Winsorize = winsor;
            end
        end

        function results = estimate(obj, Target, X)
            %ESTIMATE Estimate Scaled PCA factors
            %
            %   Input:
            %       Target: (Tx1) Independent variable (e.g., Excess Return)
            %       X: (TxN) Predictors
            %
            %   Output:
            %       results.Factors: (TxK) Extracted factors
            %       results.Loadings: (NxK) Factor loadings
            %       results.Betas: (1xN) Unisex betas from step 1

            arguments
                obj
                Target (:,1) double
                X (:,:) double
            end

            [T, N] = size(X);
            assert(length(Target) == T, 'Target and X must have same number of rows');

            % 1. Standardize X
            mu = mean(X);
            sig = std(X);
            Xs = (X - mu) ./ sig;

            % 2. Univariate Regressions
            betas = zeros(1, N);
            % Add constant for regression
            ones_T = ones(T, 1);

            % Vectorized regression might be faster?
            % beta = (x'x)^-1 x'y. Since x is just [1, xi], we can loop.
            % Or use `regress` or `\`.

            for i = 1:N
                Xi = [ones_T, Xs(:,i)];
                b = Xi \ Target;
                betas(i) = b(2);
            end

            % 3. Winsorize (Optional)
            if obj.Winsorize
                % Winsorize ABSOLUTE betas?
                % Original code: beta_win=winsor(abs(beta),[0 99]);
                % Wait, scaling usually uses raw beta, but winsorizing might be on magnitude to avoid noise?
                % The provided resource code has: % beta_win=winsor(abs(beta),[0 99]); commented out.
                % But let's support simple winsorization if requested.

                % Assuming we winsorize the betas directly if requested using 1/99 percentile.
                % Note: Implementation logic follows user preference.
                % For now, let's skip complex winsor logic unless requested.
                % If obj.Winsorize is true, we winsorize betas at LB/UB percentiles.

                p_low = prctile(betas, obj.LB);
                p_high = prctile(betas, obj.UB);
                betas(betas < p_low) = p_low;
                betas(betas > p_high) = p_high;
            end

            % 4. Scale Predictors
            Xs_scaled = Xs .* betas;

            % 5. PCA
            % SVD on Xs_scaled
            % F'F/T = I normalization usually.

            % Using MATLAB's `pca` or `svd`.
            % Resource uses `svd(y*y')` which is T x T. If N < T, better to use N x N?
            % Usually N is large in these applications.
            % Resource: [Fhat0, eigval, Fhat1] = svd(yy); fhat = Fhat0(:,1:nfac) * sqrt(bigt);

            % Let's use `pca` function for simplicity if available, or svd/eig.
            % `pca` returns coeffs (Loadings) and score (Factors).
            % [coeff, score, latent] = pca(Xs_scaled, 'NumComponents', obj.NumFactors);
            % score is normalized such to be orthogonal.

            % To match standard factor literature (F'F/T = I):
            % MATLAB `pca` score satisfies score' * score = diag(latent) * (T-1).
            % We want F'F/T = I.

            % Let's stick to the paper's/resource's method: SVD on XX' or X'X.
            % If T < N, XX' (TxT) is smaller. Use `svd(Xs_scaled * Xs_scaled')`.

            YY = Xs_scaled * Xs_scaled';
            [U, S, ~] = svd(YY, 'econ');

            % Factors F from U.
            % F = U * sqrt(T)?
            % Resource: fhat = Fhat0(:,1:nfac) * sqrt(T).

            F_hat = U(:, 1:obj.NumFactors) * sqrt(T);

            % Loadings Lambda = X' * F / T
            Lambda_hat = Xs_scaled' * F_hat / T;

            results.Factors = F_hat;
            results.Loadings = Lambda_hat;
            results.Betas = betas;
            results.EigenValues = diag(S);

            obj.Results = results;
        end
    end
end
