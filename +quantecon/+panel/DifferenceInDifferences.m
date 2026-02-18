classdef DifferenceInDifferences
    % DIFFERENCEINDIFFERENCES Difference-in-Differences Estimator
    %
    %   Usage:
    %       mdl = quantecon.panel.DifferenceInDifferences();
    %       res = mdl.estimate(y, treat, post);

    properties
        Results struct
    end

    methods
        function obj = DifferenceInDifferences()
            obj.Results = struct();
        end

        function obj = estimate(obj, y, treat, post, X, cluster)
            % ESTIMATE Estimate DiD Model
            %
            %   y_it = alpha + beta1*Treat_i + beta2*Post_t + delta*(Treat_i * Post_t) + X_it*gamma + e_it
            %
            %   Inputs:
            %       y - Outcome
            %       treat - Treatment Group Indicator (0/1)
            %       post - Post Period Indicator (0/1)
            %       X - Controls (optional)
            %       cluster - Cluster indices (optional, 1-way or 2-way)

            arguments
                obj
                y (:,1) double
                treat (:,1) double
                post (:,1) double
                X (:,:) double = []
                cluster (:,:) double = []
            end

            % Interaction term (The treatment effect)
            did_term = treat .* post;

            % Design Matrix
            if isempty(X)
                Design = [ones(size(y)), treat, post, did_term];
                ParamNames = ["Intercept", "Treat", "Post", "DiD_Effect"];
            else
                Design = [ones(size(y)), treat, post, did_term, X];
                nControls = size(X, 2);
                ParamNames = strings(1, 4 + nControls);
                ParamNames(1:4) = ["Intercept", "Treat", "Post", "DiD_Effect"];
                for k = 1:nControls
                    ParamNames(4 + k) = "Control_" + k;
                end
            end

            % OLS
            beta = (Design' * Design) \ (Design' * y);

            % Store
            obj.Results.Coefficients = beta;
            obj.Results.ParamNames = ParamNames;
            obj.Results.TreatmentEffect = beta(4);

            y_hat = Design * beta;
            resid = y - y_hat;
            obj.Results.Residuals = resid;

            if isempty(cluster)
                % Standard Errors (Homoscedastic)
                n = length(y);
                k = length(beta);
                sigma2 = sum(resid.^2) / (n - k);
                cov_beta = sigma2 * ((Design' * Design) \ eye(k));
                obj.Results.Covariance = cov_beta;
                obj.Results.SE = sqrt(diag(cov_beta));
                obj.Results.TStat = beta ./ obj.Results.SE;
                obj.Results.PValue = 2 * (1 - tcdf(abs(obj.Results.TStat), n - k));
            else
                % Cluster-Robust Standard Errors (1-way or 2-way)
                clusterRes = quantecon.panel.ClusterReg(y, Design, cluster, "HasConstant", false);
                obj.Results.Covariance = clusterRes.Covariance;
                obj.Results.SE = clusterRes.SE;
                obj.Results.TStat = clusterRes.tStat;
                obj.Results.PValue = clusterRes.pValue;
            end
        end
    end
end
