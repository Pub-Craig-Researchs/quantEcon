classdef Connectedness < handle
    %CONNECTEDNESS Diebold-Yilmaz Financial Spillover Indexes + Advance Decompositions
    %
    %   Computes Total, Directional, and Net Connectedness using:
    %   1. Rolling Window VAR
    %   2. TVP-VAR (QBLL)
    %   3. Frequency Domain (BK18)
    %   4. Quantile VAR (QVAR, White et al. 2015)
    %   5. Elastic Net VAR (LASSO/Ridge/ElasticNet)
    %
    %   Decomposition Types:
    %   - "standard": Generalized Forecast Error Variance Decomposition (Diebold-Yilmaz)
    %   - "joint": Joint Connectedness (Lastrapes & Wiesen, 2021) / Extended Joint (Balcilar et al. 2021)

    properties
        Data table          % Input Data
        Method (1,1) string {mustBeMember(Method, ["rolling", "tvp", "frequency", "qvar", "elasticnet"])} = "rolling"
        Lags (1,1) double = 2
        Horizon (1,1) double = 10
        WindowSize (1,1) double = 200

        % Results
        Model               % The underlying model object (or struct)
        FEVD (:,:,:) double % (Time, Var, Shock)
        Indices struct      % Struct with Total, To, From, Net, Pairwise
        Time (:,1)          % Time vector
    end

    methods
        function obj = Connectedness(data)
            if istimetable(data)
                obj.Data = timetable2table(data);
                obj.Time = data.Properties.RowTimes;
            elseif istable(data)
                obj.Data = data;
                obj.Time = (1:height(data))';
            else
                obj.Data = array2table(data);
                obj.Time = (1:size(data,1))';
            end
            obj.Indices = struct();
        end

        function obj = estimate(obj, opts)
            arguments
                obj
                opts.Method (1,1) string = obj.Method
                opts.Lags (1,1) double = obj.Lags
                opts.WindowSize (1,1) double = obj.WindowSize
                opts.Shrinkage (1,1) double = 0.05
                opts.Bandwidth (1,1) double = -1
                opts.Tau (1,1) double = 0.5
                opts.Alpha (1,1) double = 1        % 1: Lasso, 0: Ridge
                opts.CVFolds (1,1) double = 5
            end

            obj.Method = opts.Method;
            obj.Lags = opts.Lags;
            obj.WindowSize = opts.WindowSize;

            Y = table2array(obj.Data);

            if obj.Method == "tvp"
                fprintf('Estimating TVP-VAR (QBLL)...\n');
                mdl = quantecon.models.QbllVar(Y, 'Lags', obj.Lags, ...
                    'Shrinkage', opts.Shrinkage, 'Bandwidth', opts.Bandwidth);
                mdl.estimate();
                obj.Model = mdl;
            elseif obj.Method == "rolling"
                fprintf('Estimating Rolling VAR (Window=%d)...\n', obj.WindowSize);
                obj.Model = struct('Type', 'rolling');
            elseif obj.Method == "frequency"
                fprintf('Will estimate Frequency Domain VAR during decompose()...\n');
                obj.Model = struct('Type', 'frequency');
            elseif obj.Method == "qvar"
                fprintf('Estimating Quantile VAR (Tau=%.2f)...\n', opts.Tau);
                [B, Sigma] = quantecon.finance.Connectedness.estimate_qvar(Y, obj.Lags, opts.Tau);
                obj.Model = struct('Type', 'qvar', 'B', B, 'Sigma', Sigma);
            elseif obj.Method == "elasticnet"
                fprintf('Estimating Elastic Net VAR (Alpha=%.2f)...\n', opts.Alpha);
                [B, Sigma] = quantecon.finance.Connectedness.estimate_elasticnet(Y, obj.Lags, opts.Alpha, opts.CVFolds);
                obj.Model = struct('Type', 'elasticnet', 'B', B, 'Sigma', Sigma);
            end
        end

        function obj = decompose(obj, options)
            arguments
                obj
                options.Horizon (1,1) double = obj.Horizon
                options.DecompType (1,1) string {mustBeMember(options.DecompType, ["standard", "joint"])} = "standard"
            end
            obj.Horizon = options.Horizon;

            Y = table2array(obj.Data);
            [T, N] = size(Y);
            p = obj.Lags;
            h = obj.Horizon;
            decompType = options.DecompType;

            if obj.Method == "tvp"
                irfs = obj.Model.irf(h);
                [T_eff, ~, ~, ~] = size(irfs);
                fevd_mat = zeros(T_eff, N, N);

                if decompType == "joint"
                    error('Joint decomposition for TVP-VAR is not fully supported in this version. Use standard.');
                else
                    irf_sq = irfs(:, 2:end, :, :).^2;
                    irf_sum = squeeze(sum(irf_sq, 2));
                    for t = 1:T_eff
                        row_sums = sum(squeeze(irf_sum(t,:,:)), 2);
                        fevd_mat(t, :, :) = squeeze(irf_sum(t, :, :)) ./ row_sums;
                    end
                end
                obj.FEVD = fevd_mat;

            elseif obj.Method == "rolling"
                win = obj.WindowSize;
                T_est = T - win + 1;
                fevd_mat = zeros(T_est, N, N);

                parfor t = 1:T_est
                    subY = Y(t : t + win - 1, :);
                    [X_sub, y_vec] = quantecon.utils.VarUtils.var_setup(subY, p);
                    B = X_sub \ y_vec;
                    Resid = y_vec - X_sub * B;
                    Sigma = cov(Resid);

                    if decompType == "joint"
                        fevd_mat(t, :, :) = quantecon.finance.Connectedness.compute_joint_fevd(B, Sigma, h);
                    else
                        B_trans = B(2:end, :)';
                        irf_t = quantecon.utils.VarUtils.compute_girf(B_trans, Sigma, N, p, h);
                        num = sum(irf_t(2:end, :, :).^2, 1);
                        num = squeeze(num);
                        den = sum(num, 2);
                        fevd_mat(t, :, :) = num ./ den;
                    end
                end
                obj.FEVD = fevd_mat;

            elseif obj.Method == "frequency"
                obj.FEVD = obj.compute_bk18(Y, p, h, options);

            elseif obj.Method == "qvar" || obj.Method == "elasticnet"
                B = obj.Model.B;
                Sigma = obj.Model.Sigma;
                if decompType == "joint"
                    obj.FEVD = quantecon.finance.Connectedness.compute_joint_fevd(B, Sigma, h);
                else
                    B_trans = B(2:end, :)';
                    irf_t = quantecon.utils.VarUtils.compute_girf(B_trans, Sigma, N, p, h);
                    num = sum(irf_t(2:end, :, :).^2, 1);
                    num = squeeze(num);
                    den = sum(num, 2);
                    fevd_rep = num ./ den;
                    obj.FEVD = reshape(fevd_rep, [1, N, N]);
                end
            end

            obj.compute_indices();
        end

        function compute_indices(obj)
            fevd_data = obj.FEVD;
            [T_nodes, N_vars, ~] = size(fevd_data);
            total = zeros(T_nodes, 1);
            to_idx = zeros(T_nodes, N_vars);
            from_idx = zeros(T_nodes, N_vars);
            net_idx = zeros(T_nodes, N_vars);

            for t = 1:T_nodes
                theta = squeeze(fevd_data(t, :, :));
                off_diag = sum(theta(:)) - trace(theta);
                total(t) = off_diag / N_vars * 100;
                to_idx(t, :) = (sum(theta, 1) - diag(theta)') * 100;
                from_idx(t, :) = (sum(theta, 2)' - diag(theta)') * 100;
                net_idx(t, :) = to_idx(t, :) - from_idx(t, :);
            end

            obj.Indices.Total = total;
            obj.Indices.To = to_idx;
            obj.Indices.From = from_idx;
            obj.Indices.Net = net_idx;
            obj.Indices.Pairwise = fevd_data * 100;
        end

        function fevd = compute_bk18(~, Y, p, h, options)
            [~, N] = size(Y);
            [X, y_vec] = quantecon.utils.VarUtils.var_setup(Y, p);
            B = X \ y_vec;
            Sigma = cov(y_vec - X * B);
            A_comp = B(2:end, :)';

            if isfield(options, 'Range'); range = options.Range; else; range = [0, pi]; end
            n_grid = 100;
            omega = linspace(range(1), range(2), n_grid);
            d_omega = omega(2) - omega(1);

            numerators = zeros(N, N);
            denominators = zeros(N, 1);

            for w = omega
                ALw = eye(N);
                for l = 1:p
                    Al = A_comp(:, (l-1)*N+1 : l*N);
                    ALw = ALw - Al * exp(-1i * l * w);
                end
                Psi = inv(ALw);
                Spect = Psi * Sigma * Psi';
                PS = Psi * Sigma;
                denom_full = real(diag(Spect));

                for i = 1:N
                    for j = 1:N
                        num_w = (1/Sigma(j,j)) * abs(PS(i,j))^2;
                        numerators(i,j) = numerators(i,j) + num_w * d_omega;
                    end
                    denominators(i) = denominators(i) + denom_full(i) * d_omega;
                end
            end

            fevd_static = zeros(N, N);
            for i = 1:N
                fevd_static(i, :) = numerators(i, :) ./ denominators(i);
            end
            row_sum = sum(fevd_static, 2);
            fevd_static = fevd_static ./ row_sum;
            fevd = reshape(fevd_static, [1, N, N]);
        end
    end

    methods(Static)
        function [B, Sigma] = estimate_qvar(Y, p, tau)
            [X, y_vec] = quantecon.utils.VarUtils.var_setup(Y, p);
            [T_eff, N] = size(y_vec);
            K = size(X, 2);

            B = zeros(K, N);
            Resid = zeros(T_eff, N);

            c = 0.05 * std(y_vec(:));

            opts = optimoptions('fminunc', 'Algorithm', 'trust-region', ...
                'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective', 'Display', 'off');

            for i = 1:N
                b0 = (X'*X) \ (X'*y_vec(:, i));
                b_hat = fminunc(@(b) quantecon.finance.Connectedness.qvar_loss(b, X, y_vec(:, i), tau, c), b0, opts);
                B(:, i) = b_hat; % ensure transpose properly? b_hat is Kx1
                Resid(:, i) = y_vec(:, i) - X * b_hat;
            end
            Sigma = cov(Resid);
        end

        function [f, g, H] = qvar_loss(b, X, y, tau, c)
            u = y - X * b;
            u_c = max(min(u / c, 500), -500);
            f = sum(tau * u + c * log(1 + exp(-u_c)));

            if nargout > 1
                w = 1 ./ (1 + exp(u_c));
                g = -X' * (tau - w);
                if nargout > 2
                    W = diag(w .* (1 - w) / c);
                    H = X' * W * X;
                end
            end
        end

        function [B, Sigma] = estimate_elasticnet(Y, p, alpha, nfolds)
            [X, y_vec] = quantecon.utils.VarUtils.var_setup(Y, p);
            [T_eff, N] = size(y_vec);
            K = size(X, 2);
            B = zeros(K, N);
            Resid = zeros(T_eff, N);

            X_no_const = X(:, 2:end);
            for i = 1:N
                [B_lasso, FitInfo] = lasso(X_no_const, y_vec(:, i), ...
                    'Alpha', alpha, 'CV', nfolds, 'Standardize', true);
                idx = FitInfo.Index1SE;
                if isempty(idx), idx = 1; end
                beta = B_lasso(:, idx);
                B(1, i) = FitInfo.Intercept(idx);
                B(2:end, i) = beta;
                Resid(:, i) = y_vec(:, i) - (X_no_const * beta + FitInfo.Intercept(idx));
            end
            Sigma = cov(Resid);
        end

        function A = wold_matrices(B, N, p, h)
            A_comp = B(2:end, :)';
            A = zeros(N, N, h);
            A(:, :, 1) = eye(N);
            for i = 2:h
                Ai = zeros(N, N);
                for l = 1:min(i-1, p)
                    Al = A_comp(:, (l-1)*N+1 : l*N);
                    Ai = Ai + Al * A(:, :, i - l);
                end
                A(:, :, i) = Ai;
            end
        end

        function fevd_mat = compute_joint_fevd(B, Sigma, h)
            N = size(Sigma, 1);
            p = (size(B, 1) - 1) / N;

            A = quantecon.finance.Connectedness.wold_matrices(B, N, p, h);

            Xi = zeros(N, N);
            for step = 1:h
                Xi = Xi + A(:, :, step) * Sigma * A(:, :, step)';
            end

            IK = eye(N);
            S_jnt_from = zeros(N, 1);

            for i = 1:N
                Mi = IK(:, [1:i-1, i+1:N]);
                Sigma_i_inv = pinv(Mi' * Sigma * Mi);
                num_S = 0;
                for step = 1:h
                    Ah = A(:, :, step);
                    term1 = IK(i, :) * Ah * Sigma * Mi;
                    term2 = Mi' * Sigma * Ah' * IK(:, i);
                    num_S = num_S + term1 * Sigma_i_inv * term2;
                end
                S_jnt_from(i) = num_S / Xi(i, i);
            end

            B_trans = B(2:end, :)';
            irf_t = quantecon.utils.VarUtils.compute_girf(B_trans, Sigma, N, p, h);
            num = sum(irf_t(2:end, :, :).^2, 1);
            num = squeeze(num);
            den = sum(num, 2);
            gFEVD = num ./ den;

            gFEVD_diag = gFEVD;
            gFEVD_diag(logical(eye(N))) = 0;

            from_gFEVD = sum(gFEVD_diag, 2);
            lambda = S_jnt_from ./ from_gFEVD;

            jFEVD = gFEVD .* lambda;

            jFEVD_diag = jFEVD;
            jFEVD_diag(logical(eye(N))) = 0;
            from_jnt = sum(jFEVD_diag, 2);

            for i = 1:N
                jFEVD(i, i) = 1 - from_jnt(i);
            end

            fevd_mat = reshape(jFEVD, [1, N, N]);
        end
    end
end
