classdef SunAbraham < handle
    %SUNABRAHAM Interaction Weighted Estimator (Sun and Abraham, 2020)
    %   Estimates dynamic treatment effects robust to heterogeneous effects.
    %   Fully interacted TWFE model with cohort-time interactions.

    properties
        Cluster (1,1) string {mustBeMember(Cluster, ["Robust", "Entity", "TwoWay"])} = "Entity"
    end

    properties (SetAccess = private)
        Results
        Model
    end

    methods
        function obj = SunAbraham(varargin)
            if nargin > 0
                obj.set(varargin{:});
            end
        end

        function obj = set(obj, varargin)
            p = inputParser;
            addParameter(p, 'Cluster', obj.Cluster);
            parse(p, varargin{:});
            obj.Cluster = p.Results.Cluster;
        end

        function res = estimate(obj, Y, id, time, treatment)
            %ESTIMATE Estimate SA2020

            arguments
                obj
                Y (:,1) double
                id (:,1) double
                time (:,1) double
                treatment (:,1) double
            end

            % 1. Determine Cohorts (E_i)
            % E_i is the first time unit i is treated.
            % If never treated, E_i = Inf.

            [uid, ~, idIdx] = unique(id);
            n = length(uid);
            E = inf(n, 1);

            for i = 1:n
                mask = (idIdx == i);
                ti = time(mask);
                treat_i = treatment(mask);

                % Check if ever treated
                if any(treat_i == 1)
                    % First treatment time
                    % Assuming treatment is absorbing or we take the first
                    treated_times = ti(treat_i == 1);
                    E(i) = min(treated_times);
                end
            end

            % Map E back to full data
            E_expanded = E(idIdx);

            % 2. Clean Control Group (Never Treated or Last Treated)
            % SA2020 usually uses never-treated or last-treated as base.
            % We will use never-treated (E=Inf) as reference if available.

            % 3. Relative Time (K = t - E_i)
            K = time - E_expanded;
            K(isinf(E_expanded)) = -9999; % Sentinel

            % 4. Interaction Weighted Estimation
            % Regression: Y_it = mu_i + lambda_t + sum_{e, k} delta_{e,k} * 1(E_i=e) * 1(t-E_i=k) + epsilon
            % Only for k != -1 (reference)

            % Unique cohorts (excluding Inf)
            cohorts = unique(E(~isinf(E)));

            % Unique relative times
            rel_times = unique(K(K ~= -9999));
            rel_times = rel_times(rel_times ~= -1); % Exclude reference period

            % Build Design Matrix for Interactions
            % D_{e,k} = 1(E_i=e) * 1(K=k)

            % This can be large.
            % Using FixedEffects class to handle mu_i and lambda_t
            % Regressors X are the interaction dummies.

            % Map to construct X efficiently?
            % For large N, T, this loop is slow.
            % But SA2020 aggregates these anyway.

            % Let's use a simplified approach:
            % Regress Y on Cohort-Time dummies (excluding controls and ref period)
            % We need to saturate the model.

            % Actually, we can implement it as a specialized WOOLDRIDGE estimator
            % where the heterogeneity is specifically by Cohort.

            % Construct dummies
            nInteractions = length(cohorts) * length(rel_times);
            X = zeros(length(Y), nInteractions);
            col = 0;

            meta = struct('e', [], 'k', []);

            for e = cohorts'
                for k = rel_times'
                    col = col + 1;
                    mask = (E_expanded == e) & (K == k);
                    X(:, col) = double(mask);
                    meta.e(col) = e;
                    meta.k(col) = k;
                end
            end

            % Remove empty columns (combinations that don't exist)
            valid_cols = any(X, 1);
            X = X(:, valid_cols);
            meta.e = meta.e(valid_cols);
            meta.k = meta.k(valid_cols);

            % Estimate with TWFE
            % Y = mu_i + lambda_t + X * delta + eps
            % Estimate with TWFE
            % Y = mu_i + lambda_t + X * delta + eps
            % Note: FixedEffects does not support clustering directly in this version.
            % We use it to partial out fixed effects or just run it.
            % Be careful: FixedEffects.estimate(y, X, id, time)

            fe = quantecon.panel.FixedEffects("TimeEffects", true, "Cluster", obj.Cluster);
            fe = fe.estimate(Y, X, id, time);

            delta = fe.Results.Coefficients;
            covDelta = fe.Results.Covariance;
            se = sqrt(diag(covDelta));


            % 5. Aggregation (ATT_k)
            % SA recommends aggregating delta_{e,k} weighted by share of cohort e in relative time k

            % For each Relative Time k
            unique_k = unique(meta.k);
            att_k = zeros(length(unique_k), 1);
            att_se = zeros(length(unique_k), 1);

            for i = 1:length(unique_k)
                k_val = unique_k(i);

                % Find which deltas correspond to this k
                idx = find(meta.k == k_val);

                if isempty(idx), continue; end

                % Weights: Sample share of each cohort at this relative time?
                % S & A weights are P(E=e | E \in cohorts_that_experience_k)

                deltas_k = delta(idx);

                % Calculate weights
                counts = sum(X(:, idx), 1);
                weights = counts / sum(counts);

                att_k(i) = sum(deltas_k .* weights');

                % SE Aggregation (delta method approximation assuming indep or using Vcov)
                % We need full Vcov for correct SE.
                % FixedEffects returns specific SEs, but maybe not full Vcov.
                % Assuming indep for now or simple sum if we lack Vcov
                % var(sum(w*d)) = w' * V * w
                w = weights(:);
                att_se(i) = sqrt(w' * covDelta(idx, idx) * w);
            end

            res = struct('ATT_k', att_k, 'RelativeTimes', unique_k, 'SE', att_se, ...
                'FullDeltas', delta, 'DeltaSE', se, 'Covariance', covDelta, 'Meta', meta);
            obj.Results = res;
        end
    end
end
