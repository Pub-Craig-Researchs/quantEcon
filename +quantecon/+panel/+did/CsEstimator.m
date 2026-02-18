classdef CsEstimator
    % CSESTIMATOR Callaway & Sant'Anna (2021) Difference-in-Differences Estimator
    %
    %   Estimates Group-Time Average Treatment Effects (ATT(g,t)) allowing for
    %   heterogeneous treatment effects and dynamic adoption.
    %
    %   Usage:
    %       mdl = quantecon.panel.did.CsEstimator();
    %       mdl = mdl.estimate(y, treat, time, id);

    properties
        Delta (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        ControlGroup (1,1) string {mustBeMember(ControlGroup, ["NeverTreated", "NotYetTreated"])} = "NotYetTreated"
        BasePeriod (1,1) string {mustBeMember(BasePeriod, ["Universal", "Varying"])} = "Universal"
        Method (1,1) string {mustBeMember(Method, ["Unconditional", "OR"])} = "Unconditional"
        Results struct
    end

    methods
        function obj = CsEstimator()
            obj.Results = struct();
        end

        function obj = estimate(obj, y, treat, time, id, X)
            % ESTIMATE Estimate ATT(g,t)
            %
            % Inputs:
            %   y       - Outcome variable (N*T x 1)
            %   treat   - Treatment indicator (0/1) (N*T x 1)
            %             Note: 'treat' should be 1 if treated at time t, 0 otherwise.
            %             (Staggered adoption implies once treated, always treated)
            %   time    - Time variable (N*T x 1)
            %   id      - Individual ID (N*T x 1)
            %   X       - Covariates (optional)

            arguments
                obj
                y (:,1) double
                treat (:,1) double
                time (:,1) double
                id (:,1) double
                X (:,:) double = []
            end

            % 1. Data Setup & Group Identification
            [ids, ~, idIdx] = unique(id);
            N_units = length(ids);

            [times, ~, timeIdx] = unique(time);
            T_periods = length(times);
            timeMap = containers.Map(num2cell(times), num2cell(1:T_periods));

            % Identify Group G_i for each unit (First time treated)
            % Assuming balanced panel logic or reshaping for ease
            % Let's create a matrix representation for speed if possible, or loop

            G = inf(N_units, 1);

            % Pre-calculate sums for G
            for i = 1:N_units
                idx_i = (idIdx == i);
                t_i = time(idx_i);
                d_i = treat(idx_i);

                % Find first period where d_i == 1
                % Sort by time
                [t_i_sorted, sortPos] = sort(t_i);
                d_i_sorted = d_i(sortPos);

                first_idx = find(d_i_sorted == 1, 1, 'first');
                if ~isempty(first_idx)
                    G(i) = t_i_sorted(first_idx);
                end
            end

            % Unique groups (excluding Inf)
            g_list = unique(G(~isinf(G)));

            % 2. Compute ATT(g,t)
            % Loop over groups g and times t

            ATT_gt = [];
            Res_gt = struct('g', [], 't', [], 'att', [], 'se', [], 'N', []);

            count = 0;

            % Aggregate data to (i, t) level if needed, but assuming unique (i,t)
            % Store y in a more accessible format: y_mat(i, t_idx)
            % But panels can be unbalanced. Let's stick to vector ops with logical indexing.

            % Pre-compute means by (i, t) for performance
            % For Balanced Panel optimization:
            y_mat = NaN(N_units, T_periods);
            for i = 1:N_units
                idx_i = (idIdx == i);
                % Map times to indices
                t_vals = time(idx_i);
                t_indices = zeros(size(t_vals));
                for k=1:length(t_vals), t_indices(k) = timeMap(t_vals(k)); end
                y_vals = y(idx_i);
                y_mat(i, t_indices) = y_vals;
            end

            for g_idx = 1:length(g_list)
                g = g_list(g_idx);
                g_time_idx = timeMap(g);

                % Determine Control Group units
                if obj.ControlGroup == "NeverTreated"
                    control_units = (isinf(G));
                else % NotYetTreated
                    % Units not treated by time t (will depend on t loop)
                    % But standard CS defines NotYet as not treated by g + something?
                    % Actually CS2021 defines NotYetTreated based on current t.
                    % But simpler version: G > t_max or G > current_t
                    % Let's use the definition: Unit i is control if G_i > t + delta
                    % We'll handle this inside the t loop
                end

                for t_idx = 1:T_periods
                    t = times(t_idx);

                    % Determine Reference Period (Pre-treatment)
                    % Universal: g - 1 (or g - 1 - delta)
                    if t >= g
                        ref_time = g - 1 - obj.Delta;
                    else
                        % Pre-treatment period
                        if obj.BasePeriod == "Universal"
                            ref_time = g - 1 - obj.Delta;
                        else
                            ref_time = t - 1; % Varying base
                        end
                    end

                    if ~isKey(timeMap, ref_time), continue; end
                    ref_time_idx = timeMap(ref_time);

                    % Long Difference: Y_it - Y_i,ref
                    dy = y_mat(:, t_idx) - y_mat(:, ref_time_idx);
                    valid_dy = ~isnan(dy);

                    % Tratment Group: G_i == g
                    treated_units = (G == g) & valid_dy;

                    % Control Group
                    if obj.ControlGroup == "NeverTreated"
                        control_mask = isinf(G) & valid_dy;
                    else
                        % Not yet treated by t (and possibly g)
                        % Definition: Not treated by max(t, g)?
                        % Usually: G > t (for current t) AND G ~= g
                        control_mask = (G > max(t, g)) & (G ~= g) & valid_dy;
                    end

                    if sum(treated_units) == 0 || sum(control_mask) == 0, continue; end

                    % Data for inference
                    dy_treat = dy(treated_units);
                    dy_control = dy(control_mask);

                    if obj.Method == "Unconditional"
                        % Estimate ATT (Simple Difference in Means for unconditional)
                        att = mean(dy_treat) - mean(dy_control);

                    elseif obj.Method == "OR"
                        if isempty(X)
                            error('OR method requires covariates X.');
                        end

                        % Extract Covariates (Covariates at reference time? Or average?)
                        % CS2021 typically uses pre-treatment covariates (constant or time-varying at ref)
                        % Let's assume X is time-invariant or we take X at ref_period
                        % For simplicity in this implementation, assuming X is provided as (N*T x K)
                        % and we take X at ref_time_idx.

                        % X_ref = X(idx_t_indices(ref_time_idx), :); % This logic needs refinement for vector X
                        % Current X input is (N*T x K). We need X_i_ref.
                        % Let's refine data extraction above.

                        % REFINED LOGIC:
                        % X_mat needs to be constructed like y_mat
                        % But X might be large.
                        % Let's fallback to vector indexing.

                        % Indices in the original vectors corresponding to (unit, time)
                        % We need mapping from (u, t) -> row_index
                        % This is expensive to find every time.
                        % Constraint: For OR, let's assume balanced panel sorted by id, time for now
                        % or assume X is time-invariant and passed as (N x K).
                        % If X is (N*T x K), we need to pick the rows.

                        % Allow X to be (N x K) or (N*T x K)
                        [rx, cx] = size(X);
                        if rx == N_units
                            % Time invariant
                            X_c = X(control_mask, :);
                            X_t = X(treated_units, :);
                        elseif rx == length(y)
                            % Time varying - taking at reference period
                            % This requires finding rows for (control_units, ref_time)
                            % TODO: Implement robust indexing. For now error.
                            error('Time-varying X not yet supported in OR. Pass time-invariant X (N x K).');
                        else
                            error('X must be (N x K) or match y dimensions.');
                        end

                        % Outcome Regression
                        % 1. Regress dY on X for Control Group
                        % dy_control = X_c * beta + e
                        Design_c = [ones(sum(control_mask), 1), X_c];
                        beta = Design_c \ dy_control;

                        % 2. Predict Counterfactual for Treated
                        Design_t = [ones(sum(treated_units), 1), X_t];
                        dy_treat_hat = Design_t * beta;

                        % 3. ATT = mean(dy_treat - dy_treat_hat)
                        att = mean(dy_treat - dy_treat_hat);
                    end

                    % Store
                    count = count + 1;
                    Res_gt(count).g = g;
                    Res_gt(count).t = t;
                    Res_gt(count).att = att;
                    Res_gt(count).N = sum(treated_units) + sum(control_mask);

                    % TODO: Influence function for SE
                end
            end

            obj.Results.ATT_gt = Res_gt;
        end

        function res = aggregate(obj, type)
            % AGGREGATE Aggregate results
            arguments
                obj
                type (1,1) string = "EventStudy"
            end

            if isempty(obj.Results) || ~isfield(obj.Results, 'ATT_gt')
                error('Model not estimated.');
            end

            res = quantecon.panel.did.Aggregator.aggregate(obj.Results.ATT_gt, type);
        end
    end
end
