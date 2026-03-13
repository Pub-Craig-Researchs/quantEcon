classdef BaconDecomposition
    % BACONDECOMPOSITION Goodman-Bacon (2021) TWFE Decomposition
    %
    %   Decomposes the standard TWFE estimator into weighted averages of
    %   2x2 DiD comparisons. Includes Treated-vs-NeverTreated,
    %   Early-vs-Late, and Late-vs-Early components.
    %
    %   Reference:
    %       Goodman-Bacon (2021), "Difference-in-Differences with Variation
    %       in Treatment Timing", JoE 225(2).

    methods (Static)
        function res = decompose(y, treat, time, id)
            % DECOMPOSE Perform Bacon decomposition
            arguments
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
            end

            % 1. Overall TWFE Beta
            mdl = quantecon.panel.FixedEffects();
            res_fe = mdl.estimate(y, treat, id, time);
            beta_twfe = res_fe.Results.Coefficients;

            % 2. Identify Groups (first treatment time per unit)
            [ids, ~, idIdx] = unique(id);
            N = length(ids);
            times_all = unique(time);
            T = length(times_all);
            t_min = min(times_all);
            t_max = max(times_all);

            G = inf(N, 1);
            for i = 1:N
                idx_i = (idIdx == i);
                t_i = time(idx_i);
                d_i = treat(idx_i);
                [t_sorted, sp] = sort(t_i);
                d_sorted = d_i(sp);
                first = find(d_sorted == 1, 1, 'first');
                if ~isempty(first)
                    G(i) = t_sorted(first);
                end
            end

            g_list = unique(G(~isinf(G)));
            n_g = length(g_list);
            has_never = any(isinf(G));

            % Group shares (fraction of units)
            n_total = N;
            n_share = zeros(n_g + 1, 1);  % last entry = never-treated
            for j = 1:n_g
                n_share(j) = sum(G == g_list(j)) / n_total;
            end
            n_share(end) = sum(isinf(G)) / n_total;

            % 3. Variance of within-transformed D for the full sample
            % V_D = Var(D_tilde) where D_tilde is within-transformed D
            D_mean_i = accumarray(idIdx, treat) ./ accumarray(idIdx, 1);
            D_tilde = treat - D_mean_i(idIdx);
            VarD = var(D_tilde, 1); %#ok<NASGU> used for reference in weight normalization

            % 4. Iterate pairs
            maxPairs = n_g * (n_g - 1) + n_g;  % upper bound
            pairs = repmat(struct('weight', 0, 'beta', 0, 'type', "", ...
                'k', 0, 'l', 0), maxPairs, 1);
            nPair = 0;

            for k_idx = 1:n_g
                k = g_list(k_idx);
                n_k = n_share(k_idx);

                % Fraction of time treated for group k
                Dk_bar = max(0, (t_max - k + 1)) / T;  % post-periods / total

                % --- Treated vs Never Treated ---
                if has_never
                    n_u = n_share(end);
                    [beta_ku, valid_ku] = quantecon.panel.did.BaconDecomposition.did_2x2_est( ...
                        y, time, idIdx, G, k, Inf, t_min, t_max);
                    if valid_ku
                        % Weight: (n_k + n_u)^2 * Dk_bar * (1-Dk_bar) / VarD
                        w_ku = (n_k + n_u)^2 * Dk_bar * (1 - Dk_bar);
                        nPair = nPair + 1;
                        pairs(nPair) = struct('weight', w_ku, 'beta', beta_ku, ...
                            'type', "TreatedVsNever", 'k', k, 'l', Inf);
                    end
                end

                % --- Early vs Late and Late vs Early ---
                for l_idx = 1:n_g
                    if k_idx == l_idx, continue; end
                    l = g_list(l_idx);
                    if k >= l, continue; end  % only process k < l once

                    n_l = n_share(l_idx);

                    % Early (k) vs Late (l): control = l, period restricted to < l
                    Dk_mid = max(0, (l - k)) / (l - t_min);  % treated fraction in mid period
                    if (l - t_min) <= 0, Dk_mid = 0; end

                    [beta_el, valid_el] = quantecon.panel.did.BaconDecomposition.did_2x2_est( ...
                        y, time, idIdx, G, k, l, t_min, t_max);
                    if valid_el && Dk_mid > 0 && Dk_mid < 1
                        w_el = (n_k + n_l)^2 * Dk_mid * (1 - Dk_mid);
                        nPair = nPair + 1;
                        pairs(nPair) = struct('weight', w_el, 'beta', beta_el, ...
                            'type', "EarlyVsLate", 'k', k, 'l', l);
                    end

                    % Late (l) vs Early (k): control = k (already treated), period >= k
                    Dl_late = max(0, (t_max - l + 1)) / (t_max - k + 1);
                    if (t_max - k + 1) <= 0, Dl_late = 0; end

                    [beta_le, valid_le] = quantecon.panel.did.BaconDecomposition.did_2x2_le( ...
                        y, time, idIdx, G, l, k, t_min, t_max);
                    if valid_le && Dl_late > 0 && Dl_late < 1
                        w_le = (n_k + n_l)^2 * Dl_late * (1 - Dl_late);
                        nPair = nPair + 1;
                        pairs(nPair) = struct('weight', w_le, 'beta', beta_le, ...
                            'type', "LateVsEarly", 'k', l, 'l', k);
                    end
                end
            end
            pairs = pairs(1:nPair);

            % Normalize weights to sum to 1
            total_w = sum([pairs.weight]);
            if total_w > 0
                for j = 1:nPair
                    pairs(j).weight = pairs(j).weight / total_w;
                end
            end

            % Verify: weighted sum should approximate beta_twfe
            beta_check = sum([pairs.weight] .* [pairs.beta]);

            res = struct('Beta_TWFE', beta_twfe, 'Beta_Decomposed', beta_check, ...
                'Pairs', pairs, 'NPairs', nPair);
        end

        function [beta, valid] = did_2x2_est(y, time, idIdx, G, treated_g, control_g, ~, ~)
            % DID_2X2_EST  Standard 2x2 DiD: early(k) vs late/never(l)
            %   Restricts to periods < control_g (if finite) and units in k or l.

            mask_units = (G(idIdx) == treated_g) | (G(idIdx) == control_g);
            if ~isinf(control_g)
                mask_time = (time < control_g);
            else
                mask_time = true(size(time));
            end
            mask = mask_units & mask_time;

            if sum(mask) < 4
                beta = 0; valid = false; return;
            end

            y_sub = y(mask);
            t_sub = time(mask);
            post = (t_sub >= treated_g);
            is_treat = (G(idIdx(mask)) == treated_g);

            cells = [sum(is_treat & post), sum(is_treat & ~post), ...
                     sum(~is_treat & post), sum(~is_treat & ~post)];
            if any(cells == 0)
                beta = 0; valid = false; return;
            end

            y11 = mean(y_sub(is_treat & post));
            y10 = mean(y_sub(is_treat & ~post));
            y01 = mean(y_sub(~is_treat & post));
            y00 = mean(y_sub(~is_treat & ~post));

            beta = (y11 - y10) - (y01 - y00);
            valid = ~isnan(beta);
        end

        function [beta, valid] = did_2x2_le(y, time, idIdx, G, late_g, early_g, ~, ~)
            % DID_2X2_LE  Late-vs-Early 2x2 DiD
            %   Treatment = late_g, Control = early_g (already treated).
            %   Restricts to periods >= early_g. Post = (t >= late_g).

            mask_units = (G(idIdx) == late_g) | (G(idIdx) == early_g);
            mask_time = (time >= early_g);
            mask = mask_units & mask_time;

            if sum(mask) < 4
                beta = 0; valid = false; return;
            end

            y_sub = y(mask);
            t_sub = time(mask);
            post = (t_sub >= late_g);
            is_treat = (G(idIdx(mask)) == late_g);

            cells = [sum(is_treat & post), sum(is_treat & ~post), ...
                     sum(~is_treat & post), sum(~is_treat & ~post)];
            if any(cells == 0)
                beta = 0; valid = false; return;
            end

            y11 = mean(y_sub(is_treat & post));
            y10 = mean(y_sub(is_treat & ~post));
            y01 = mean(y_sub(~is_treat & post));
            y00 = mean(y_sub(~is_treat & ~post));

            beta = (y11 - y10) - (y01 - y00);
            valid = ~isnan(beta);
        end
    end
end
