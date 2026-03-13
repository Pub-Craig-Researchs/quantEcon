classdef WooldridgeEstimator
    % WOOLDRIDGEESTIMATOR Wooldridge (2021) Two-Way Mundlak Estimator
    %
    %   Estimates ATT(g,t) using pooled OLS with cohort-time interactions.
    %   Efficient standard errors via clustering.
    %
    %   Usage:
    %       mdl = quantecon.panel.did.WooldridgeEstimator();
    %       res = mdl.estimate(y, treat, time, id);

    properties
        Results struct
    end

    methods
        function obj = WooldridgeEstimator()
            obj.Results = struct();
        end

        function res = estimate(~, y, treat, time, id, X)
            % ESTIMATE Estimate DiD via Wooldridge (2021)
            arguments
                ~
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
                X (:,:) double = []
            end

            % 1. Data Setup
            [ids, ~, idIdx] = unique(id);
            N_units = length(ids);
            [times, ~, timeIdx] = unique(time);

            % Identify Groups (First Treatment Time)
            G = inf(N_units, 1);
            for i = 1:N_units
                idx_i = (idIdx == i);
                t_i = time(idx_i);
                d_i = treat(idx_i);
                % Sort by time to correctly find first treatment period
                [t_i_sorted, sortPos] = sort(t_i);
                d_i_sorted = d_i(sortPos);
                first_idx = find(d_i_sorted == 1, 1, 'first');
                if ~isempty(first_idx)
                    G(i) = t_i_sorted(first_idx);
                end
            end
            G_long = G(idIdx);

            % Cohort Dummies & Interactions
            % Interaction: 1(G_i = g) * 1(t = s)
            % We need to expand this into a design matrix of dummies

            % Groups present in data (excluding Inf)
            g_list = unique(G(~isinf(G)));

            % Design Matrix Z construction
            % Z includes:
            % - Unit FEs? No, Wooldridge uses Mundlak device or just Group + Time FEs if balanced?
            % Wooldridge (2021) proposes:
            % y_it = alpha + mu_g + lambda_t + sum_{g} sum_{t} tau_{gt} * 1(G_i=g)*1(period=t) + e_it
            % This is exactly the TWFE specification but saturated.
            % It allows recovering ATT(g,t).

            % Regressors:
            % 1. Intercept
            % 2. Group FEs (Time Invariant) -> 1(G_i = g)
            % 3. Time FEs (Unit Invariant)  -> 1(period = t)
            % 4. Interactions (ATTs)        -> 1(G_i = g) * 1(period = t) [For t >= g]

            % Note: Base levels to avoid collinearity
            % - Drop one Group FE (e.g., Control or First Group)
            % - Drop one Time FE (e.g., First Period)
            % - Interactions: Only for Treated (t >= g). If we include all, some are collinear.
            %   Actually we specifically want the coefficients on 1(G=g, t=t).
            %   Standard TWFE saturates everything.

            % Let's build the matrix matrix Z row by row or via sparse construction.

            % Time Dummies (T-1)
            time_dummies = dummyvar(timeIdx);
            % Drop first
            time_dummies = time_dummies(:, 2:end);

            % Group Dummies (G-1)
            % Map G to 1..ng
            [~, ~, G_idx] = unique(G_long); % Includes Inf as a group
            group_dummies = dummyvar(G_idx);
            % Drop one (e.g. Never Treated if exists, or first group)
            % Let's drop the first column
            group_dummies = group_dummies(:, 2:end);

            % Interactions
            % Pre-allocate interaction columns
            maxCols = length(g_list) * length(times);
            interaction_cols = zeros(length(y), maxCols);
            interaction_names = repmat(struct('g', 0, 't', 0), maxCols, 1);
            nInt = 0;

            for g_val = g_list'
                for t_val = times'
                    if t_val >= g_val
                        nInt = nInt + 1;
                        interaction_cols(:, nInt) = double(G_long == g_val & time == t_val);
                        interaction_names(nInt) = struct('g', g_val, 't', t_val);
                    end
                end
            end
            interaction_cols = interaction_cols(:, 1:nInt);
            interaction_names = interaction_names(1:nInt);

            Z = [ones(length(y), 1), group_dummies, time_dummies, interaction_cols, X];

            % OLS
            beta = Z \ y;

            % Residuals
            u = y - Z * beta;

            % Cluster-Robust SE (cluster by id)
            clusterRes = quantecon.panel.ClusterReg(y, Z, id, "HasConstant", false);

            % Extract Coefficients of Interest (Interactions)
            num_bases = 1 + size(group_dummies, 2) + size(time_dummies, 2);
            num_interactions = size(interaction_cols, 2);

            att_betas = beta(num_bases+1 : num_bases+num_interactions);
            att_se = clusterRes.SE(num_bases+1 : num_bases+num_interactions);

            % Store Results in Struct Array
            Res_gt = struct('g', [], 't', [], 'att', [], 'se', [], 'N', []);
            % N needs to be calculated per cell

            for k = 1:num_interactions
                info = interaction_names(k);
                Res_gt(k).g = info.g;
                Res_gt(k).t = info.t;
                Res_gt(k).att = att_betas(k);
                Res_gt(k).se = att_se(k);

                % Count
                mask = (G_long == info.g & time == info.t);
                Res_gt(k).N = sum(mask);
            end

            res = struct();
            res.ATT_gt = Res_gt;
            res.Beta = beta;
            res.SE = clusterRes.SE;
            res.tStat = clusterRes.tStat;
            res.pValue = clusterRes.pValue;
            res.Covariance = clusterRes.Covariance;
            res.Residuals = u;
        end

        function res = aggregate(~, res_struct, type)
            % Delegate to Aggregator
            if nargin < 3, type = "Simple"; end
            res = quantecon.panel.did.Aggregator.aggregate(res_struct.ATT_gt, type);
        end
    end
end
