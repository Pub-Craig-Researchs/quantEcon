classdef Aggregator
    % AGGREGATOR Aggregation logic for DiD Estimators
    %
    %   Aggregates ATT(g,t) into Event Study, Group, or Calendar Time effects.

    methods (Static)
        function res = aggregate(att_gt_struct, type, weights)
            % AGGREGATE Aggregate ATT(g,t)
            %
            % Inputs:
            %   att_gt_struct - Struct array with fields: g, t, att, N, (se)
            %   type          - "EventStudy", "Group", "Calendar", "Simple"
            %   weights       - "GroupSize" (default), "Equal"

            arguments
                att_gt_struct struct
                type (1,1) string {mustBeMember(type, ["Simple", "EventStudy", "Group", "Calendar"])} = "Simple"
                weights (1,1) string {mustBeMember(weights, ["GroupSize", "Equal"])} = "GroupSize"
            end

            % Extract vectors
            g = [att_gt_struct.g]';
            t = [att_gt_struct.t]';
            att = [att_gt_struct.att]';
            N = [att_gt_struct.N]';

            % Calculate Weights
            if weights == "Equal"
                w = ones(size(N));
            else
                w = N; % Weight by cell size
            end

            switch type
                case "Simple"
                    % Simple weighted average of all post-treatment ATTs (t >= g)
                    mask = (t >= g);
                    if ~any(mask)
                        res = struct('Estimate', NaN);
                        return;
                    end

                    est = sum(att(mask) .* w(mask)) / sum(w(mask));
                    res = struct('Type', 'Simple', 'Estimate', est);

                case "EventStudy"
                    % Aggregate by relative time e = t - g
                    e = t - g;
                    u_e = unique(e);

                    estimates = zeros(length(u_e), 1);
                    for k = 1:length(u_e)
                        ek = u_e(k);
                        mask = (e == ek);
                        estimates(k) = sum(att(mask) .* w(mask)) / sum(w(mask));
                    end

                    res = table(u_e, estimates, 'VariableNames', {'EventTime', 'Estimate'});

                case "Group"
                    % Aggregate by Group g (Average over t >= g)
                    u_g = unique(g);
                    estimates = zeros(length(u_g), 1);

                    for k = 1:length(u_g)
                        gk = u_g(k);
                        mask = (g == gk) & (t >= gk);
                        if ~any(mask)
                            estimates(k) = NaN;
                        else
                            estimates(k) = sum(att(mask) .* w(mask)) / sum(w(mask));
                        end
                    end

                    res = table(u_g, estimates, 'VariableNames', {'Group', 'Estimate'});

                case "Calendar"
                    % Aggregate by Time t (Average over g <= t)
                    u_t = unique(t);
                    estimates = zeros(length(u_t), 1);

                    for k = 1:length(u_t)
                        tk = u_t(k);
                        mask = (t == tk) & (g <= tk); % Only treated cohorts
                        if ~any(mask)
                            estimates(k) = NaN;
                        else
                            estimates(k) = sum(att(mask) .* w(mask)) / sum(w(mask));
                        end
                    end

                    res = table(u_t, estimates, 'VariableNames', {'Time', 'Estimate'});
            end
        end
    end
end
