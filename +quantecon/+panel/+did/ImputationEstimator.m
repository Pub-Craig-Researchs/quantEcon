classdef ImputationEstimator
    % IMPUTATIONESTIMATOR Borusyak, Jaravel, Spiess (2024) Estimator
    %
    %   Estimates ATT by imputing counterfactual outcomes for treated units
    %   using a model fitted on untreated observations (D=0).
    %
    %   Usage:
    %       mdl = quantecon.panel.did.ImputationEstimator();
    %       res = mdl.estimate(y, treat, time, id, X);
    %
    %   Reference:
    %       Borusyak, Jaravel & Spiess (2024), "Revisiting Event-Study
    %       Designs: Robust and Efficient Estimation", ReStud.

    methods
        function res = estimate(~, y, treat, time, id, X)
            arguments
                ~
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
                X (:,:) double = []
            end

            % 1. Identify estimation samples
            mask_0 = (treat == 0);
            mask_1 = (treat == 1);
            N1 = sum(mask_1);

            if sum(mask_0) == 0
                error('No untreated observations found.');
            end
            if N1 == 0
                error('No treated observations found.');
            end

            % 2. Fit outcome model on D=0 using covariates (if any)
            beta_hat = zeros(0, 1);
            if ~isempty(X)
                fe_mdl = quantecon.panel.FixedEffects();
                res_0 = fe_mdl.estimate(y(mask_0), X(mask_0, :), id(mask_0), time(mask_0));
                beta_hat = res_0.Results.Coefficients;
            end

            % 3. Partial out covariates: Y_star = y - X*beta
            if isempty(X)
                Y_star = y;
            else
                Y_star = y - X * beta_hat;
            end

            % 4. Recover unit and time FEs from D=0 sample
            [ids, ~, idIdx_full] = unique(id);
            [times, ~, tIdx_full] = unique(time);
            N_units = length(ids);
            T_periods = length(times);

            id_0 = idIdx_full(mask_0);
            t_0  = tIdx_full(mask_0);
            y_star_0 = Y_star(mask_0);
            n0 = length(y_star_0); %#ok<NASGU>

            % Solve for FEs using iterative demeaning (Alternating Projections)
            % to avoid the rank-deficiency issue of the dummy approach.
            % Initialize: alpha_i = mean(y_star_0) for each unit i
            alpha = zeros(N_units, 1);
            lambda = zeros(T_periods, 1);

            % Compute means via accumarray on D=0 sample
            for iter = 1:100
                % Update lambda: lambda_t = mean_t(y_star_0 - alpha_i)
                resid_alpha = y_star_0 - alpha(id_0);
                cnt_t = accumarray(t_0, 1, [T_periods, 1]);
                sum_t = accumarray(t_0, resid_alpha, [T_periods, 1]);
                active_t = (cnt_t > 0);
                lambda(active_t) = sum_t(active_t) ./ cnt_t(active_t);

                % Update alpha: alpha_i = mean_i(y_star_0 - lambda_t)
                resid_lambda = y_star_0 - lambda(t_0);
                cnt_i = accumarray(id_0, 1, [N_units, 1]);
                sum_i = accumarray(id_0, resid_lambda, [N_units, 1]);
                active_i = (cnt_i > 0);
                alpha(active_i) = sum_i(active_i) ./ cnt_i(active_i);

                % Check convergence
                fitted = alpha(id_0) + lambda(t_0);
                resid = y_star_0 - fitted;
                if iter > 1 && max(abs(resid - prev_resid)) < 1e-12
                    break;
                end
                prev_resid = resid;
            end

            % 5. Impute Y(0) for treated observations
            id_1 = idIdx_full(mask_1);
            t_1  = tIdx_full(mask_1);

            y0_hat = alpha(id_1) + lambda(t_1);
            if ~isempty(X)
                y0_hat = y0_hat + X(mask_1, :) * beta_hat;
            end

            % 6. Individual treatment effects and ATT
            y_1 = y(mask_1);
            tau_it = y_1 - y0_hat;
            att = mean(tau_it);

            % 7. Standard errors (cluster by unit)
            % V(ATT) = (1/N1^2) * sum_i (sum_{t: D_it=1} tau_it - ATT)^2
            % This is the BJS heteroskedasticity-robust SE clustered by unit
            uid_treated = unique(id_1);
            nC = length(uid_treated);
            score_sum = 0;
            for c = 1:nC
                mask_c = (id_1 == uid_treated(c));
                score_c = sum(tau_it(mask_c) - att);
                score_sum = score_sum + score_c^2;
            end

            % Small-sample correction: G/(G-1)
            if nC > 1
                adj = nC / (nC - 1);
            else
                adj = 1;
            end
            se_att = sqrt(adj * score_sum / N1^2);

            % 8. Event-study decomposition (ATT by relative time)
            % Identify group (first treatment) per unit
            G_unit = inf(N_units, 1);
            for i = 1:N_units
                idx_i = (idIdx_full == i);
                t_i = time(idx_i);
                d_i = treat(idx_i);
                [ts, sp] = sort(t_i);
                ds = d_i(sp);
                fi = find(ds == 1, 1, 'first');
                if ~isempty(fi)
                    G_unit(i) = ts(fi);
                end
            end

            % Relative time for treated obs
            rel_time = time(mask_1) - G_unit(id_1);
            u_rel = unique(rel_time);
            n_rel = length(u_rel);
            att_e = zeros(n_rel, 1);
            se_e  = zeros(n_rel, 1);

            for j = 1:n_rel
                emask = (rel_time == u_rel(j));
                att_e(j) = mean(tau_it(emask));

                % Cluster SE for this event time
                score_sum_e = 0;
                n1_e = sum(emask);
                for c = 1:nC
                    mask_ce = (id_1 == uid_treated(c)) & emask;
                    if any(mask_ce)
                        score_sum_e = score_sum_e + (sum(tau_it(mask_ce) - att_e(j)))^2;
                    end
                end
                if nC > 1
                    se_e(j) = sqrt(nC / (nC - 1) * score_sum_e / n1_e^2);
                end
            end

            res = struct('ATT', att, 'SE', se_att, 'Tau_it', tau_it, ...
                'Beta', beta_hat, 'Alpha', alpha, 'Lambda', lambda, ...
                'EventStudy', table(u_rel, att_e, se_e, ...
                    'VariableNames', {'RelTime', 'ATT', 'SE'}));
        end
    end
end
