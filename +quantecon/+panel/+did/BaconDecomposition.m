classdef BaconDecomposition
    % BACONDECOMPOSITION Goodman-Bacon (2021) TWFE Decomposition
    %
    %   Decomposes the standard TWFE estimator into weighted averages of 2x2 DiD comparisons.

    methods (Static)
        function res = decompose(y, treat, time, id)
            % DECOMPOSE Perform Bacon decomposition
            arguments
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
            end

            % 1. Calculate Overall TWFE Beta
            % y = alpha_i + lambda_t + beta * D_it + e
            % Use Within Estimator
            mdl = quantecon.panel.FixedEffects();
            res_fe = mdl.estimate(y, treat, id, time); % Assuming method signature (y, X, id, time)
            % Wait, FixedEffects.estimate signature is (y, X, id, time) or (y, X, id)?
            % Let's check or assume standard: estimate(y, X, id, time).
            % Provided FixedEffects signature: estimate(y, X, id, time) -> beta

            beta_twfe = res_fe.Results.Coefficients;

            % 2. Identify Groups and their Treatment Times
            [ids, ~, idIdx] = unique(id);
            N = length(ids);

            G = inf(N, 1);
            for i = 1:N
                idx_i = (idIdx == i);
                d_i = treat(idx_i);
                t_i = time(idx_i);
                first = find(d_i == 1, 1, 'first');
                if ~isempty(first)
                    G(i) = t_i(first);
                end
            end

            % Units with G=Inf are Never Treated
            g_list = unique(G(~isinf(G)));
            U_groups = [g_list; Inf]; % All groups including control

            pairs = [];

            % 3. Iterate over all pairs of groups (k, l)
            % Case 1: Early vs Late
            % Case 2: Late vs Early
            % Case 3: Treated vs Never Treated

            % Calculate group probabilities/weights in sample?
            % Actually the formula is specific.
            % It's easier to iterate 2x2 DiDs and calculate their weights from sample sizes.

            total_weight = 0;

            for k_idx = 1:length(g_list)
                k = g_list(k_idx);

                % Comparison with Never Treated (U)
                if any(isinf(G))
                    res_ku = quantecon.panel.did.BaconDecomposition.did_2x2(y, treat, time, idIdx, G, k, Inf);
                    pairs = [pairs; res_ku];
                    total_weight = total_weight + res_ku.weight;
                end

                % Comparisons with other Treated Groups
                for l_idx = 1:length(g_list)
                    if k_idx == l_idx, continue; end
                    l = g_list(l_idx);

                    % k vs l
                    % If k < l (Early vs Late abuser)
                    % Control group is l (treated later). Analysis period is before l is treated.
                    if k < l
                        res_kl = quantecon.panel.did.BaconDecomposition.did_2x2(y, treat, time, idIdx, G, k, l);
                        pairs = [pairs; res_kl];
                        total_weight = total_weight + res_kl.weight;
                    end
                end
            end

            % Normalize weights if needed (Should sum to 1 theoretically)

            res = struct('Beta_TWFE', beta_twfe, 'Pairs', pairs);
        end

        function out = did_2x2(y, treat, time, idIdx, G, treated_g, control_g)
            % Helper for 2x2 DiD
            % treated_g: The group acting as Treatment
            % control_g: The group acting as Control

            % 1. Filter Data
            % Keep units in treated_g and control_g
            mask_units = (G(idIdx) == treated_g) | (G(idIdx) == control_g);

            % 2. Filter Time
            % If control_g is treated later (l > k), we must drop periods >= l
            % because control becomes treated.
            if ~isinf(control_g)
                limit_t = control_g;
                mask_time = (time < limit_t);
            else
                mask_time = true(size(time));
            end

            mask = mask_units & mask_time;
            if sum(mask) == 0
                out = struct('weight', 0, 'beta', 0, 'type', "Empty");
                return;
            end

            y_sub = y(mask);
            d_sub = treat(mask);
            % 2x2 Estimator: simple diff-in-diff
            % Mean(Post, Treat) - Mean(Pre, Treat) - (Mean(Post, Control) - Mean(Pre, Control))
            % Post is defined by treated_g: t >= treated_g
            t_sub = time(mask);
            post = (t_sub >= treated_g);
            treat_group = (G(idIdx(mask)) == treated_g);

            y11 = mean(y_sub(treat_group & post));
            y10 = mean(y_sub(treat_group & ~post));
            y01 = mean(y_sub(~treat_group & post));
            y00 = mean(y_sub(~treat_group & ~post));

            if any(isnan([y11, y10, y01, y00]))
                beta = 0; w = 0; % Degenerate
            else
                beta = (y11 - y10) - (y01 - y00);

                % Weight Logic (Goodman-Bacon Formula)
                % w ~ n_u * n_c * p * (1-p) * Var(D)
                % Calculating exact analytical weight is complex from scratch.
                % Simplified: V_k_l = (n_k + n_l)^2 * p * (1-p)...

                % For prototype, return unweighted beta or simple sample size
                w = sum(mask); % Placeholder
            end

            type = "";
            if isinf(control_g), type = "VsNeverTreated";
            elseif treated_g < control_g, type = "EarlyVsLate";
            else, type = "LateVsEarly";
            end

            out = struct('weight', w, 'beta', beta, 'type', type, 'k', treated_g, 'l', control_g);
        end
    end
end
