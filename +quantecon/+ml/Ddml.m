classdef Ddml < handle
    %DDML Double/Debiased Machine Learning (Chernozhukov et al. 2018)
    %   Partially Linear Regression via cross-fitting and ML nuisance estimation.
    %     Y = D*theta + g(X) + epsilon
    %     D = m(X) + v
    %
    %   Features:
    %     - Cross-fitted nuisance (g, m) with Lasso/Ridge/Random Forest
    %     - Lasso uses CV-selected lambda
    %     - VCE: HC1, 1-way cluster, 2-way cluster (Cameron, Gelbach & Miller 2011)
    %
    %   Usage:
    %     ddml = quantecon.ml.Ddml('ModelY','lasso','ModelD','lasso');
    %     res  = ddml.estimate(Y, D, X);
    %     fprintf('theta = %.4f (SE = %.4f)\n', res.Coefficients, res.Se);
    %
    %     % 1-way cluster SE
    %     ddml = quantecon.ml.Ddml('VCEType','cluster');
    %     res  = ddml.estimate(Y, D, X, 'ClusterId', firmId);
    %
    %     % 2-way cluster SE (e.g., firm × time)
    %     ddml = quantecon.ml.Ddml('VCEType','twoway');
    %     res  = ddml.estimate(Y, D, X, 'ClusterId', [firmId, timeId]);

    properties
        ModelY (1,1) string {mustBeMember(ModelY,["lasso","ridge","rf"])} = "lasso"
        ModelD (1,1) string {mustBeMember(ModelD,["lasso","ridge","rf"])} = "lasso"
        KFolds (1,1) double {mustBeInteger, mustBePositive} = 5
        AddConstant (1,1) logical = true
        VCEType (1,1) string {mustBeMember(VCEType,["hc1","cluster","twoway"])} = "hc1"
    end

    properties (SetAccess = private)
        Coef
        Se
        Results
    end

    methods
        function obj = Ddml(varargin)
            if nargin > 0
                obj.set(varargin{:});
            end
        end

        function obj = set(obj, varargin)
            p = inputParser;
            addParameter(p, 'ModelY', obj.ModelY);
            addParameter(p, 'ModelD', obj.ModelD);
            addParameter(p, 'KFolds', obj.KFolds);
            addParameter(p, 'AddConstant', obj.AddConstant);
            addParameter(p, 'VCEType', obj.VCEType);
            parse(p, varargin{:});
            obj.ModelY = p.Results.ModelY;
            obj.ModelD = p.Results.ModelD;
            obj.KFolds = p.Results.KFolds;
            obj.AddConstant = p.Results.AddConstant;
            obj.VCEType = p.Results.VCEType;
        end

        function res = estimate(obj, Y, D, X, varargin)
            %ESTIMATE  Estimate PLR via cross-fitting.
            %   Y : (N,1) outcome
            %   D : (N,d) treatment(s)
            %   X : (N,p) controls / confounders
            %   Optional Name-Value:
            %     'ClusterId' : (N,1) for 1-way cluster, (N,2) for 2-way cluster

            ip = inputParser;
            addRequired(ip, 'Y');
            addRequired(ip, 'D');
            addRequired(ip, 'X');
            addParameter(ip, 'ClusterId', []);
            parse(ip, Y, D, X, varargin{:});
            clusterId = ip.Results.ClusterId;

            Y = Y(:);
            N  = length(Y);
            dD = size(D, 2);

            % validate cluster inputs
            if obj.VCEType ~= "hc1"
                assert(~isempty(clusterId), ...
                    'Ddml:ClusterRequired', ...
                    'ClusterId required for VCEType = "%s".', obj.VCEType);
                assert(size(clusterId, 1) == N, ...
                    'Ddml:ClusterSize', 'ClusterId must have N rows.');
                if obj.VCEType == "twoway"
                    assert(size(clusterId, 2) == 2, ...
                        'Ddml:TwowayCols', ...
                        'ClusterId must have 2 columns for twoway clustering.');
                end
            end

            cv = cvpartition(N, 'KFold', obj.KFolds);

            Y_tilde = zeros(N, 1);
            D_tilde = zeros(N, dD);

            % --- cross-fitting ---
            for k = 1:obj.KFolds
                trn = cv.training(k);
                tst = cv.test(k);

                % nuisance g(X) = E[Y|X]
                mdl_y  = obj.trainModel(X(trn,:), Y(trn), obj.ModelY);
                y_hat  = obj.predictModel(mdl_y, X(tst,:), obj.ModelY);
                Y_tilde(tst) = Y(tst) - y_hat;

                % nuisance m_j(X) = E[D_j|X]  for each treatment column
                for j = 1:dD
                    mdl_d  = obj.trainModel(X(trn,:), D(trn,j), obj.ModelD);
                    d_hat  = obj.predictModel(mdl_d, X(tst,:), obj.ModelD);
                    D_tilde(tst,j) = D(tst,j) - d_hat;
                end
            end

            % --- final OLS on residuals ---
            if obj.AddConstant
                Z = [ones(N,1), D_tilde];
            else
                Z = D_tilde;
            end
            kZ = size(Z, 2);

            b = (Z' * Z) \ (Z' * Y_tilde);         % OLS
            e = Y_tilde - Z * b;                     % residuals

            % --- Variance estimation ---
            XX_inv = (Z' * Z) \ eye(kZ);

            switch obj.VCEType
                case "hc1"
                    % HC1 robust standard errors
                    score = Z .* e;
                    meat  = score' * score;
                    V     = (N / (N - kZ)) * (XX_inv * meat * XX_inv);

                case "cluster"
                    % 1-way cluster-robust (Cameron & Miller 2015)
                    V = obj.clusterMeat(Z, e, clusterId(:,1), kZ, N, XX_inv);

                case "twoway"
                    % 2-way cluster-robust (Cameron, Gelbach & Miller 2011)
                    %   V = V_1 + V_2 - V_12
                    V1  = obj.clusterMeat(Z, e, clusterId(:,1), kZ, N, XX_inv);
                    V2  = obj.clusterMeat(Z, e, clusterId(:,2), kZ, N, XX_inv);
                    % intersection cluster
                    c12 = clusterId(:,1) * 1e8 + clusterId(:,2);  % combine
                    V12 = obj.clusterMeat(Z, e, c12, kZ, N, XX_inv);
                    V   = V1 + V2 - V12;
            end

            se_all = sqrt(max(diag(V), 0));   % guard against tiny negatives

            % extract theta (drop intercept if present)
            if obj.AddConstant
                theta = b(2:end);
                se    = se_all(2:end);
            else
                theta = b;
                se    = se_all;
            end

            tstat = theta ./ se;
            pval  = 2 * (1 - normcdf(abs(tstat)));

            obj.Coef = theta;
            obj.Se   = se;

            res = struct('Coefficients', theta, ...
                         'Se',           se, ...
                         'tstat',        tstat, ...
                         'pval',         pval, ...
                         'VCEType',      char(obj.VCEType), ...
                         'Residuals',    e, ...
                         'Y_tilde',      Y_tilde, ...
                         'D_tilde',      D_tilde);
            obj.Results = res;
        end
    end

    methods (Access = private)
        function V = clusterMeat(~, Z, e, cid, kZ, N, XX_inv)
            %CLUSTERMEAT  Cluster-robust variance for one clustering dimension.
            %   Cameron & Miller (2015): V = G/(G-1)*(N-1)/(N-k) * (X'X)^{-1} B (X'X)^{-1}
            %   where B = sum_g (sum_i_in_g Z_i*e_i)' * (sum_i_in_g Z_i*e_i)

            [uGroups, ~, gIdx] = unique(cid);
            G    = numel(uGroups);
            meat = zeros(kZ, kZ);

            score = Z .* e;                 % N x kZ  individual scores
            for g = 1:G
                sg   = sum(score(gIdx == g, :), 1);   % 1 x kZ  cluster score
                meat = meat + sg' * sg;
            end

            dof  = (G / (G - 1)) * ((N - 1) / (N - kZ));
            V    = dof * (XX_inv * meat * XX_inv);
        end

        function mdl = trainModel(~, X, y, type)
            switch type
                case "lasso"
                    if isempty(X) || size(X,2) == 0
                        mdl = struct('type','const','val',mean(y));
                    else
                        % CV-selected lambda
                        [B, FitInfo] = lasso(X, y, 'CV', 5);
                        best = FitInfo.IndexMinMSE;
                        mdl  = struct('type','lasso', ...
                                      'B', B(:,best), ...
                                      'Intercept', FitInfo.Intercept(best));
                    end
                case "ridge"
                    if isempty(X) || size(X,2) == 0
                        mdl = struct('type','const','val',mean(y));
                    else
                        % ridge with scaled lambda grid via GCV
                        B   = ridge(y, X, 0.1, 0);
                        mdl = struct('type','ridge','B', B);
                    end
                case "rf"
                    if isempty(X) || size(X,2) == 0
                        mdl = struct('type','const','val',mean(y));
                    else
                        mdl = fitrensemble(X, y, 'Method', 'Bag', ...
                            'NumLearningCycles', 100);
                    end
            end
        end

        function yp = predictModel(~, mdl, X, type)
            % handle constant fallback
            if isstruct(mdl) && isfield(mdl,'type') && mdl.type == "const"
                yp = repmat(mdl.val, size(X,1), 1);
                return
            end
            switch type
                case "lasso"
                    yp = X * mdl.B + mdl.Intercept;
                case "ridge"
                    yp = [ones(size(X,1),1), X] * mdl.B;
                case "rf"
                    yp = predict(mdl, X);
            end
        end
    end
end
