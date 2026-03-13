classdef QbllVar < handle
    %QBLLVAR Quasi-Bayesian Local Likelihood VAR (Petrova 2019)
    %
    %   Implements Time-Varying Parameter VAR using Kernel-weighted Local Likelihood.
    %   Reference: Petrova, K. (2019). "A Quasi-Bayesian Local Likelihood Approach to
    %   Time-Varying Parameter VAR Models". Journal of Econometrics.
    %
    %   Usage:
    %       mdl = quantecon.models.QbllVar(Y, 'Lags', 2, 'Shrinkage', 0.05);
    %       mdl = mdl.estimate();
    %       irf = mdl.irf(10);

    properties
        Y (:,:) double          % Data (T x N)
        Lags (1,1) double {mustBeInteger, mustBePositive} = 2
        Shrinkage (1,1) double {mustBePositive} = 0.05
        Bandwidth (1,1) double = -1 % Default: sqrt(T)

        % Estimation Results
        Beta (:,:,:) double     % (T_eff x K x N) ? Or (T_eff x K*N)?
        % Structure in resource: bayesb is K*N vector per time
        % Let's store as (T_eff, K, N) for ease.
        Sigma (:,:,:) double    % (T_eff, N, N) Covariance matrices

        IsEstimated (1,1) logical = false
    end

    properties (Access = private)
        X (:,:) double          % Design Matrix (T_eff x K)
        Y_eff (:,:) double      % Effective Y (T_eff x N)
        T_eff (1,1) double      % Effective Sample Size
        N (1,1) double          % Number of Variables
        K (1,1) double          % Number of Regressors per equation (N*p + 1)
        Prior (1,1) struct      % Prior hyperparams (SI, PI, a, RI)
    end

    methods
        function obj = QbllVar(Data, options)
            %QBLLVAR Constructor
            arguments
                Data (:,:) double
                options.Lags (1,1) double = 2
                options.Shrinkage (1,1) double = 0.05
                options.Bandwidth (1,1) double = -1
            end

            obj.Y = Data;
            obj.Lags = options.Lags;
            obj.Shrinkage = options.Shrinkage;
            obj.Bandwidth = options.Bandwidth;
        end

        function obj = estimate(obj)
            %ESTIMATE Run QBLL Estimation

            [T_raw, n_vars] = size(obj.Y);
            p = obj.Lags;

            % 1. Prepare Data (VAR Coeffs)
            % Lagged Design Matrix
            % X: [1, y(t-1)', ..., y(t-p)']
            [X_mat, Y_vec] = quantecon.utils.VarUtils.var_setup(obj.Y, p);

            obj.Y_eff = Y_vec;
            obj.X = X_mat;
            obj.T_eff = size(Y_vec, 1);
            obj.N = n_vars;
            obj.K = n_vars * p + 1;

            % 2. Bandwidth
            if obj.Bandwidth <= 0
                h = sqrt(obj.T_eff);
            else
                h = obj.Bandwidth;
            end

            % 3. Priors (Minnesota + Wishart)
            [SI, PI, a, RI] = obj.minn_prior(obj.Y, obj.T_eff, obj.N, p, obj.Shrinkage);

            % Sparse optimization
            priorprec0 = sparse(diag(1./diag(PI))); % Precision (inverse variance)
            SI_sparse = sparse(SI);

            % 4. Kernel Weights
            weights = quantecon.utils.VarUtils.norm_kernel(obj.T_eff, h);

            % 5. Estimation Loop (Parallelizable)
            % Storage
            B_store = zeros(obj.T_eff, obj.K, obj.N); % Store as matrix K x N per time t
            S_store = zeros(obj.T_eff, obj.N, obj.N);

            % Check for Parallel Pool
            % if isempty(gcp('nocreate')), parpool; end

            % Pre-compute constants
            XT = obj.X';

            parfor t = 1:obj.T_eff
                w = weights(t, :)'; % Weights for time t

                % Local precision
                % W = diag(w). Too big for large T. w is (T x 1).
                % X' * W * X -> Weighted sum of outer products x_s * x_s'
                % X is T x K. w is T x 1.
                % Equivalent to (X .* sqrt(w))' * (X .* sqrt(w))

                X_w = obj.X .* sqrt(w);
                y_w = obj.Y_eff .* sqrt(w);

                bayesprec = priorprec0 + X_w' * X_w;
                % bayessv = inv(bayesprec); % Invert KxK matrix

                % Mean
                % BB = bayessv * (X_w' * y_w + priorprec0 * SI_sparse);
                % Solve linear system instead of explicit inv for stability?
                RHS = (X_w' * y_w + priorprec0 * SI_sparse);
                BB = bayesprec \ RHS; % (K x N)

                % Covariance (Sigma)
                % Posterior parameters for Wishart
                % alpha_post = a + sum(w);
                % gamma_post = RI + (y-XB)'W(y-XB) ... but formula in code is more complex efficient

                % Efficient computation from resource:
                % g1 = SI' * priorprec0 * SI;
                % g2 = y' * W * y;
                % g3 = BB' * bayesprec * BB;
                % gamma = RI + g1 + g2 - g3;

                g1 = SI_sparse' * priorprec0 * SI_sparse;
                % g2: y' * diag(w) * y = (y.*sqrt(w))' * (y.*sqrt(w))
                g2 = y_w' * y_w;
                g3 = BB' * bayesprec * BB;

                bayesgamma = RI + g1 + g2 - g3;
                bayesgamma = 0.5 * (bayesgamma + bayesgamma'); % Symmetry

                % Posterior Mean of Sigma (Expected Value of Inverse Wishart)
                % E[Sigma] = gamma / (alpha - N - 1)
                alpha_post = a + sum(w);
                Sigma_post = bayesgamma / (alpha_post - obj.N - 1);

                B_store(t, :, :) = BB;
                S_store(t, :, :) = Sigma_post;
            end

            obj.Beta = B_store;
            obj.Sigma = S_store;
            obj.IsEstimated = true;
        end

        function irfs = irf(obj, horizon)
            %IRF Compute Generalized Impulse Response Functions
            %   Returns: (T_eff x Horizon x N x N)

            if ~obj.IsEstimated
                error('Model not estimated. Run estimate() first.');
            end

            h = horizon;
            T = obj.T_eff;
            n = obj.N;

            irfs = zeros(T, h+1, n, n); % Horizon+1 for step 0

            % Compute IRF for each time period t
            % No parfor here to avoid nested if already parallel? Or use parfor if estimate not running.

            B_all = obj.Beta;
            S_all = obj.Sigma;
            L = obj.Lags;

            parfor t = 1:T
                Bt = squeeze(B_all(t, :, :)); % (K x N)
                St = squeeze(S_all(t, :, :)); % (N x N)

                % Companion Form
                % Coefficients in Bt are [Const; Lag1_var1; Lag1_var2... LagL_varN]
                % Reorder to Companion [A1, A2, ..., Ap]

                % Bt is (K x N). Const is row 1.
                % Equation i: y_i = c + A1 y(t-1) ...
                % Beta structure from var_setup:
                % X = [1, y(t-1), ...]
                % Coeffs for eq i are column i of Bt.

                % Extract A matrices
                % A = [A1, A2, ..., Ap] (N x N*p)
                A_comp = Bt(2:end, :)'; % (N*p x N)' -> (N x N*p)

                % GIRF
                ir_t = quantecon.utils.VarUtils.compute_girf(A_comp, St, n, L, h);
                irfs(t, :, :, :) = ir_t;
            end
        end
    end

    methods (Access = private, Static)
        function [SI, PI, a, RI] = minn_prior(Y, T_eff, N, L, shrinkage)
            % MINN_PRIOR adapted from Petrova 2018
            % Simplified logic based on resource

            % 1. Prior Mean (SI): Random Walk / White Noise
            % Assume data is stationary or diffed?
            % Resource sets 0.5 for lag 1 (Persistent but stationary)
            % Structure: (K x N)
            K = N*L + 1;
            SI = zeros(K, N);

            % Set Lag 1 diagonal to 0.5 (or 1 for RW)
            % Rows 2 to N+1 correspond to Lag 1
            for i = 1:N
                SI(1 + i, i) = 0.5;
            end

            % 2. Prior Variance (PI) - Diagonal KxK
            % Minn Prior: Tight on distant lags
            % Sigma_sq from OLS AR(1)
            sigma_sq = zeros(N, 1);
            for i = 1:N
                % Simple AR(1) OLS
                y_i = Y(L+1:end, i);
                x_i = [ones(length(y_i), 1), Y(1:end-L, i)]; % Approx
                % Resource Minn_NWprior.m has correct OLS loop
                % Let's use simplified OLS on effective sample

                % Using raw data Y passed in (full sample)
                yy = Y(L+1:end, i);
                ylag = Y(L:end-1, i);
                res = yy - [ones(size(yy,1),1), ylag] * ([ones(size(yy,1),1), ylag] \ yy);
                sigma_sq(i) = var(res);
            end

            PI = zeros(K, 1);
            PI(1) = 100; % Constant loose

            % Lags
            s = 1./sigma_sq; % ? Resource: s=sigma_sq.^(-1)
            % But standard Minn prior scales by sigma_i / sigma_j
            % Resource PI formula: (shrinkage^2)*s / (lag^2)
            % Wait, s is N vector. PI is K vector.
            % (2+N*(ii-1) : 1+N*ii) are the N coeffs for lag ii

            for l = 1:L
                idx_start = 2 + (l-1)*N;
                idx_end = 1 + l*N;
                % Resource: PI(...) = (shrinkage^2) * s / (l^2)
                % This assumes dependent variable variance scaling?
                % Actually, standard is sigma_i^2 / sigma_j^2.
                % If we assume normalized data or simplified, this works.
                PI(idx_start:idx_end) = (shrinkage^2) .* (1./sigma_sq) / (l^2);
                % Wait, standard minn check:
                % Var(beta_ij,l) = (lambda/l)^2 * (sigma_i / sigma_j)^2
                % Resource implementation seems specific. I will follow resource exactly.
                % Resource: s = sigma_sq.^(-1).
                % PI entries = shrinkage^2 * s / l^2.
            end

            PI = diag(PI);

            % 3. Wishart Priors
            a = N + 2; % df
            RI = (a - N - 1) * diag(sigma_sq); % Scale matrix
        end
    end
end
