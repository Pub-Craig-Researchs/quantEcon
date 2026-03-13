classdef DccMidas < handle
    %DCCMIDAS Dynamic Conditional Correlation MIDAS Model
    %
    %   Decomposes the conditional covariance matrix into variances and
    %   the correlation matrix using a two-step estimation strategy.
    %
    %   Reference: Colacito, Engle and Ghysels (2011).

    properties
        Period (1,1) {mustBeInteger, mustBePositive} = 22
        NumLagsCorr (1,1) {mustBeInteger, mustBePositive} = 10
        NumLagsVar (1,1) {mustBeInteger, mustBePositive} = 10
        Results % Struct for results
    end

    methods
        function obj = DccMidas(varargin)
            p = inputParser;
            addParameter(p, 'Period', 22);
            addParameter(p, 'NumLagsCorr', 10);
            addParameter(p, 'NumLagsVar', 10);
            parse(p, varargin{:});
            obj.Period = p.Results.Period;
            obj.NumLagsCorr = p.Results.NumLagsCorr;
            obj.NumLagsVar = p.Results.NumLagsVar;
        end

        function res = estimate(obj, Data)
            % Data: T x N matrix of returns
            res = struct();
            [T, N] = size(Data);

            % Step 1: Univariate GARCH-MIDAS for each series
            fprintf('Step 1: Estimating univariate GARCH-MIDAS models...\n');
            H = zeros(T, N);
            res_step1 = cell(N, 1);
            std_resid = zeros(T, N);

            for i = 1:N
                mdl = quantecon.timeseries.GarchMidas('Period', obj.Period, 'NumLags', obj.NumLagsVar);
                res_step1{i} = mdl.estimate(Data(:,i));
                H(:,i) = res_step1{i}.ConditionalVariance;
                std_resid(:,i) = (Data(:,i) - res_step1{i}.Parameters(1)) ./ sqrt(H(:,i));
            end

            % Step 2: Estimate Correlation Parameters (a, b, w)
            fprintf('Step 2: Estimating correlation parameters...\n');

            % Precompute historical correlations
            num_periods = ceil(T / obj.Period);
            hist_corr = zeros(num_periods, N*N);
            for i = 1:num_periods
                idx = ((i-1)*obj.Period + 1) : min(i*obj.Period, T);
                C = corr(std_resid(idx, :));
                hist_corr(i, :) = C(:)';
            end

            % Params: [a, b, w]
            initial_guess = [0.05; 0.9; 5];
            lb = [1e-6; 1e-6; 1.001];
            ub = [0.999; 0.999; 50];

            % Stability: a + b < 1
            A = [1, 1, 0];
            b_val = 0.999;

            objective = @(p) -sum(obj.log_likelihood_corr(p, std_resid, hist_corr));

            options = optimoptions('fmincon', 'Display', 'off');
            corr_params = fmincon(objective, initial_guess, A, b_val, [], [], lb, ub, [], options);

            [ll, R, Rt] = obj.log_likelihood_corr(corr_params, std_resid, hist_corr);

            res.Step1 = res_step1;
            res.Step2Parameters = corr_params;
            res.LogLikelihoodCorr = sum(ll);
            res.CorrelationMatrices = R;
            res.LongRunCorrelationMatrices = Rt;
            res.ConditionalVariances = H;

            obj.Results = res;
        end

        function [logL, R_series, Rt_series] = log_likelihood_corr(obj, p, std_resid, hist_corr)
            [T, N] = size(std_resid);
            a = p(1);
            b = p(2);
            w_par = p(3);
            K = obj.NumLagsCorr;
            m_agg = obj.Period;

            num_periods = size(hist_corr, 1);
            weights = obj.midas_beta_weights(K, 1, w_par);

            % Long-run correlation Rt (constant over period)
            Rt = zeros(N, N, num_periods);
            for i = (K+1):num_periods
                Rt_vals = weights' * hist_corr(i-1:-1:i-K, :);
                Rt_mat = reshape(Rt_vals, N, N);
                % Normalize to ensure valid correlation matrix
                d = sqrt(diag(Rt_mat));
                Rt(:,:,i) = Rt_mat ./ (d * d');
            end
            % Fill initial
            avg_corr = reshape(mean(hist_corr, 1), N, N);
            for i = 1:K
                Rt(:,:,i) = avg_corr;
            end

            % DCC recursion
            % Q_t = (1-a-b) * Rt + a * (u_{t-1} u_{t-1}') + b * Q_{t-1}
            Q = eye(N);
            R_series = zeros(N, N, T);
            Rt_series = zeros(N, N, T);
            logL = zeros(T, 1);

            intercept = 1 - a - b;

            for t = 1:T
                p_idx = ceil(t / m_agg);
                Rt_curr = Rt(:,:,p_idx);
                Rt_series(:,:,t) = Rt_curr;

                if t > 1
                    u_prev = std_resid(t-1, :)';
                    Q = intercept * Rt_curr + a * (u_prev * u_prev') + b * Q;
                end

                % Standardize Q to get R
                d = sqrt(diag(Q));
                R = Q ./ (d * d');
                R_series(:,:,t) = R;

                % LL for correlation part
                logL(t) = -0.5 * (log(det(R)) + std_resid(t,:) * (R \ std_resid(t, :)'));
            end

            % Skip initial
            logL(1:(K*m_agg)) = 0;
        end
    end

    methods (Access = private)
        function w = midas_beta_weights(~, nlag, p1, p2)
            seq = (1:nlag)';
            u = (nlag - seq + 1) / nlag;
            w_raw = (1 - u + eps).^(p1-1) .* (u).^(p2-1);
            w = w_raw / sum(w_raw);
        end
    end
end
