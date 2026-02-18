classdef SyntheticDid
    % SYNTHETICDID Arkhangelsky et al. (2021) Synthetic DiD Estimator
    %
    %   Combines SC unit weights and Time weights to estimate ATT.

    methods
        function res = estimate(obj, Y, D)
            % ESTIMATE Estimate SDID
            % Inputs:
            %   Y: (N x T) Outcome Matrix
            %   D: (N x T) Treatment Matrix (0/1)
            %
            % Assumes Block Design: Last N_tr units are treated, Last T_post periods are post.

            [N, T] = size(Y);

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

            Y_pre_trt_avg = mean(Y_tr_pre, 1); % (1 x T_pre)

            % Weighted Treatment Post
            att_tr_post = mean(mean(Y_tr_post)); % Unweighted usually?

            % Weighted Treatment Pre (Time Weighted)
            % Y_tr_pre is (N_tr x T_pre)
            % We use lambda to weight time
            att_tr_pre = mean(Y_tr_pre * lambda(times_pre));

            % Weighted Control Post (Unit Weighted)
            % Y_co_post is (N_co x T_post)
            % We use omega to weight units
            att_co_post = mean(omega' * Y_co_post);

            % Weighted Control Pre (Unit & Time Weighted)
            % Y_co_pre is (N_co x T_pre)
            att_co_pre = omega' * Y_co_pre * lambda(times_pre);

            tau = (att_tr_post - att_tr_pre) - (att_co_post - att_co_pre);

            res = struct('ATT', tau, 'Omega', omega, 'Lambda', lambda);
        end
    end
end
