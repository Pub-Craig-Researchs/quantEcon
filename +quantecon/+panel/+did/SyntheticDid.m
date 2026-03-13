classdef SyntheticDid
    % SYNTHETICDID Arkhangelsky et al. (2021) Synthetic DiD Estimator
    %
    %   Combines SC unit weights and Time weights to estimate ATT.
    %   Includes jackknife standard errors.
    %
    %   Reference:
    %       Arkhangelsky et al. (2021), "Synthetic Difference-in-Differences",
    %       AER 111(12).

    methods
        function res = estimate(~, Y, D)
            % ESTIMATE Estimate SDID
            % Inputs:
            %   Y: (N x T) Outcome Matrix
            %   D: (N x T) Treatment Matrix (0/1)
            %
            % Assumes Block Design: treatment indicator defines blocks.

            arguments
                ~
                Y (:,:) double
                D (:,:) double
            end

            [N, T] = size(Y);
            assert(isequal(size(D), [N, T]), 'Y and D must have the same dimensions.');

            % Identify Treated and Control Blocks
            % Simplified for block design.
            % Find first treated unit
            is_treated = any(D, 2);
            units_tr = find(is_treated);
            units_co = find(~is_treated);

            % Find first treated time
            is_post_col = any(D, 1);
            times_post = find(is_post_col);
            times_pre = find(~is_post_col);

            Y_tr = Y(units_tr, :);
            Y_co = Y(units_co, :);

            Y_tr_pre = Y_tr(:, times_pre);
            Y_tr_post = Y_tr(:, times_post);
            Y_co_pre = Y_co(:, times_pre);
            Y_co_post = Y_co(:, times_post);

            % 1. Unit Weights (Omega)
            % Match Pre-trends of Treated Average using Control Units
            target_unit = mean(Y_tr_pre, 1)'; % (T_pre x 1)
            X_unit = Y_co_pre';               % (T_pre x N_co)

            omega = quantecon.optimization.Simplex.solve(X_unit, target_unit, 1e-6);

            % 2. Time Weights (Lambda)
            % Match Pre-trends of Control Average using Pre-Periods
            % We want to reweight pre-periods to match post-period levels (for constant gap).
            % Target: Average of Control in Post
            % Regressors: Control in Pre

            % The following block is replaced by a more general lambda calculation
            % target_time = mean(Y_co_post, 2); % (N_co x 1) -> Average over T_post
            % X_time = Y_co_pre;                % (N_co x T_pre)
            % lambda = quantecon.optimization.Simplex.solve(X_time, target_time, 1e-6);

            % Assuming is_pre, is_post, T_post, zeta are defined elsewhere or derived from times_pre/times_post
            % For this context, let's derive them from the existing variables:
            is_pre = false(1, T);
            is_pre(times_pre) = true;
            is_post = false(1, T);
            is_post(times_post) = true;
            T_post = length(times_post);
            zeta = 1e-6; % Assuming a default regularization parameter

            if isempty(times_pre) % If no pre-periods, use simple average for post
                lambda = zeros(T, 1);
                lambda(is_post) = 1/T_post;
            else
                % SDID Time Weights
                % Target: Constant 1 (intercept)? Or
                % "Find lambda_pre such that Y_co(pre)' * lambda_pre approx Y_co(post)' * 1/T_post"

                % Y_co is (N_co x T).
                % Y_co_pre = Y_co(:, is_pre)  (N_co x T_pre)
                % Y_co_post = Y_co(:, is_post) (N_co x T_post)

                Y_post_mean = mean(Y_co(:, is_post), 2); % (N_co x 1)

                % A = Y_co(:, is_pre)'  (T_pre x N_co)? No.
                % We want lambda (T_pre x 1) such that:
                % Y_co(:, is_pre) * lambda approx Y_post_mean

                % A = Y_co(:, is_pre)  (N_co x T_pre)
                % b = Y_post_mean      (N_co x 1)

                % Regress Y_post_mean on Y_co_pre
                % Wait, Simplex expects X*w = target
                % X = Y_co(:, is_pre)
                % target = Y_post_mean

                lambda_pre = quantecon.optimization.Simplex.solve(Y_co(:, is_pre), Y_post_mean, zeta);

                lambda = zeros(T, 1); % Lambda should be T x 1
                lambda(is_pre)  = lambda_pre;
                lambda(is_post) = 1/T_post;
            end

            % 3. Estimate ATT
            % The original ATT calculation is replaced by a call to a static method.
            % We need to prepare the inputs for the static method.
            % Assuming the static method expects:
            % Y_pre_trt_avg: (1 x T_pre) average of treated units in pre-periods
            % Y_pre_co: (N_co x T_pre) control units in pre-periods
            % Y_co: (N_co x T) full control units outcome
            % Y_trt: (N_tr x T) full treated units outcome
            % is_pre: (1 x T) logical array for pre-periods
            % is_post: (1 x T) logical array for post-periods
            % Weights, Regularization, Solver: properties of the object

            Y_pre_trt_avg = mean(Y_tr_pre, 1); %#ok<NASGU>

            % --- Compute ATT using SDID weighted DiD ---
            % tau = [lambda_post' * Y_bar_tr_post - lambda_pre' * Y_bar_tr_pre]
            %     - [lambda_post' * (omega' * Y_co_post) - lambda_pre' * (omega' * Y_co_pre)]
            % where Y_bar_tr = mean across treated units

            lambda_pre_vec  = lambda(times_pre);
            lambda_post_vec = lambda(times_post);

            % Mean across treated units
            Y_tr_bar_pre  = mean(Y_tr_pre, 1)';   % (T_pre x 1)
            Y_tr_bar_post = mean(Y_tr_post, 1)';  % (T_post x 1)

            % Weighted control outcomes
            Y_co_w_pre  = Y_co_pre' * omega;    % (T_pre x 1)
            Y_co_w_post = Y_co_post' * omega;   % (T_post x 1)

            tau = (lambda_post_vec' * Y_tr_bar_post - lambda_pre_vec' * Y_tr_bar_pre) ...
                - (lambda_post_vec' * Y_co_w_post - lambda_pre_vec' * Y_co_w_pre);

            % --- Jackknife SE (leave-one-unit-out) ---
            N_tr = length(units_tr);
            N_co = length(units_co);
            tau_jack = zeros(N_tr + N_co, 1);

            for jj = 1:(N_tr + N_co)
                if jj <= N_co
                    % Drop one control unit
                    idx_co_j = [1:jj-1, jj+1:N_co];
                    Y_co_j = Y_co(idx_co_j, :);
                    Y_tr_j = Y_tr;

                    % Re-estimate omega
                    target_j = mean(Y_tr_j(:, times_pre), 1)';
                    X_j = Y_co_j(:, times_pre)';
                    omega_j = quantecon.optimization.Simplex.solve(X_j, target_j, 1e-6);

                    Y_co_w_pre_j  = Y_co_j(:, times_pre)' * omega_j;
                    Y_co_w_post_j = Y_co_j(:, times_post)' * omega_j;

                    Y_tr_bar_pre_j  = mean(Y_tr_j(:, times_pre), 1)';
                    Y_tr_bar_post_j = mean(Y_tr_j(:, times_post), 1)';
                else
                    % Drop one treated unit
                    idx_tr_j = jj - N_co;
                    sel = [1:idx_tr_j-1, idx_tr_j+1:N_tr];
                    Y_tr_j = Y_tr(sel, :);

                    Y_co_w_pre_j  = Y_co_w_pre;
                    Y_co_w_post_j = Y_co_w_post;

                    Y_tr_bar_pre_j  = mean(Y_tr_j(:, times_pre), 1)';
                    Y_tr_bar_post_j = mean(Y_tr_j(:, times_post), 1)';
                end

                tau_jack(jj) = (lambda_post_vec' * Y_tr_bar_post_j - lambda_pre_vec' * Y_tr_bar_pre_j) ...
                             - (lambda_post_vec' * Y_co_w_post_j  - lambda_pre_vec' * Y_co_w_pre_j);
            end

            % Jackknife variance
            n_jack = length(tau_jack);
            se_tau = sqrt((n_jack - 1) / n_jack * sum((tau_jack - mean(tau_jack)).^2));

            res = struct('ATT', tau, 'SE', se_tau, 'Omega', omega, 'Lambda', lambda);
        end
    end
end
