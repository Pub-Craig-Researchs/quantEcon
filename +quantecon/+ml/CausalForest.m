classdef CausalForest < handle
    %CAUSALFOREST Causal Forest for Heterogeneous Treatment Effects
    %   Estimates CATE(x) = E[Y(1)-Y(0)|X=x] using five meta-learners:
    %
    %     "tlearner"  - Two separate outcome models (Kunzel et al. 2019)
    %     "xlearner"  - Imputed treatment effects (Kunzel et al. 2019)
    %     "honest"    - Honest Causal Forest (Athey & Imbens 2016, Wager & Athey 2018)
    %     "drlearner" - Doubly-Robust Learner (Kennedy 2023)
    %     "qte"       - Quantile Treatment Effects via quantile forests
    %
    %   Usage:
    %     cf = quantecon.ml.CausalForest('Estimator','honest','NumTrees',500);
    %     res = cf.estimate(X, Y, W);
    %     % res.CATE  (N,1)  individual treatment effects
    %     % res.ATE   scalar average treatment effect
    %
    %     % Quantile Treatment Effects
    %     cf = quantecon.ml.CausalForest('Estimator','qte','Quantiles',[.1 .25 .5 .75 .9]);
    %     res = cf.estimate(X, Y, W);
    %     % res.CQTE  (N,nQ)  conditional quantile treatment effects
    %     % res.QTE   (1,nQ)  unconditional quantile treatment effects

    properties
        Estimator (1,1) string {mustBeMember(Estimator, ...
            ["tlearner","xlearner","honest","drlearner","qte"])} = "tlearner"
        NumTrees      (1,1) double {mustBeInteger, mustBePositive} = 100
        MinLeafSize   (1,1) double {mustBeInteger, mustBePositive} = 5
        Mtry          (:,1) double = []
        MaxDepth      (1,1) double {mustBeInteger, mustBePositive} = 10
        Honesty       (1,1) logical = true
        HonestFraction    (1,1) double = 0.5
        SubsampleFraction (1,1) double = 0.5
        PropClip      (1,1) double = 0.01   % propensity clipping for DR
        Quantiles     (:,1) double = [0.10; 0.25; 0.50; 0.75; 0.90]  % for QTE
    end

    properties (SetAccess = private)
        Results
        Model
    end

    methods
        function obj = CausalForest(varargin)
            if nargin > 0
                obj.set(varargin{:});
            end
        end

        function obj = set(obj, varargin)
            p = inputParser;
            addParameter(p, 'Estimator', obj.Estimator);
            addParameter(p, 'NumTrees', obj.NumTrees);
            addParameter(p, 'MinLeafSize', obj.MinLeafSize);
            addParameter(p, 'Mtry', obj.Mtry);
            addParameter(p, 'MaxDepth', obj.MaxDepth);
            addParameter(p, 'Honesty', obj.Honesty);
            addParameter(p, 'HonestFraction', obj.HonestFraction);
            addParameter(p, 'SubsampleFraction', obj.SubsampleFraction);
            addParameter(p, 'PropClip', obj.PropClip);
            addParameter(p, 'Quantiles', obj.Quantiles);
            parse(p, varargin{:});
            flds = fieldnames(p.Results);
            for i = 1:numel(flds)
                obj.(flds{i}) = p.Results.(flds{i});
            end
        end

        function res = estimate(obj, X, Y, W)
            %ESTIMATE  Estimate CATE(x).
            %   X : (N,p) covariates
            %   Y : (N,1) outcome
            %   W : (N,1) binary treatment (0/1)

            arguments
                obj
                X (:,:) double
                Y (:,1) double
                W (:,1) double
            end

            if isempty(obj.Mtry)
                obj.Mtry = max(1, floor(sqrt(size(X,2))));
            end

            switch obj.Estimator
                case "tlearner",    res = obj.estimateTlearner(X, Y, W);
                case "xlearner",    res = obj.estimateXlearner(X, Y, W);
                case "honest",      res = obj.estimateHonest(X, Y, W);
                case "drlearner",   res = obj.estimateDrLearner(X, Y, W);
                case "qte",         res = obj.estimateQte(X, Y, W);
            end
            obj.Results = res;
        end
    end

    % ================================================================
    %  PRIVATE — meta-learner implementations
    % ================================================================
    methods (Access = private)

        % ---- T-Learner ----
        function res = estimateTlearner(obj, X, Y, W)
            learner = templateTree('MinLeafSize', obj.MinLeafSize, ...
                                   'NumVariablesToSample', obj.Mtry);
            m1 = fitrensemble(X(W==1,:), Y(W==1), 'Method','Bag', ...
                 'NumLearningCycles', obj.NumTrees, 'Learners', learner);
            m0 = fitrensemble(X(W==0,:), Y(W==0), 'Method','Bag', ...
                 'NumLearningCycles', obj.NumTrees, 'Learners', learner);
            cate = predict(m1, X) - predict(m0, X);
            res  = struct('CATE', cate, 'ATE', mean(cate));
        end

        % ---- X-Learner ----
        function res = estimateXlearner(obj, X, Y, W)
            learner = templateTree('MinLeafSize', obj.MinLeafSize, ...
                                   'NumVariablesToSample', obj.Mtry);
            args = {'Method','Bag','NumLearningCycles',obj.NumTrees,'Learners',learner};
            m1 = fitrensemble(X(W==1,:), Y(W==1), args{:});
            m0 = fitrensemble(X(W==0,:), Y(W==0), args{:});
            mu1 = predict(m1, X);  mu0 = predict(m0, X);
            D1 = Y(W==1) - mu0(W==1);   % imputed effects for treated
            D0 = mu1(W==0) - Y(W==0);   % imputed effects for control
            mt1 = fitrensemble(X(W==1,:), D1, args{:});
            mt0 = fitrensemble(X(W==0,:), D0, args{:});
            tau1 = predict(mt1, X);  tau0 = predict(mt0, X);
            prop = mean(W);
            cate = prop * tau0 + (1 - prop) * tau1;
            res  = struct('CATE', cate, 'ATE', mean(cate));
        end

        % ---- Honest Causal Forest (Athey & Imbens 2016) ----
        function res = estimateHonest(obj, X, Y, W)
            [N, p] = size(X);
            cateSum = zeros(N, 1);

            nSub = max(round(obj.SubsampleFraction * N), 2*obj.MinLeafSize);

            for b = 1:obj.NumTrees
                % 1. subsample
                idx = randsample(N, nSub, false);

                % 2. split into build / estimation
                nBuild = max(round(obj.HonestFraction * nSub), obj.MinLeafSize);
                nEst   = nSub - nBuild;
                if nEst < obj.MinLeafSize; nBuild = nSub - obj.MinLeafSize; end
                perm   = randperm(nSub);
                bIdx   = idx(perm(1:nBuild));
                eIdx   = idx(perm(nBuild+1:end));

                % 3. build causal tree on build sample
                tree = obj.buildCausalTree( ...
                    X(bIdx,:), Y(bIdx), W(bIdx), ...
                    obj.MinLeafSize, obj.Mtry, p, obj.MaxDepth);

                % 4. honest re-estimation on estimation sample
                tree = obj.honestReestimate( ...
                    tree, X(eIdx,:), Y(eIdx), W(eIdx));

                % 5. predict CATE for all N observations
                cateSum = cateSum + obj.predictTree(tree, X);
            end

            cate = cateSum / obj.NumTrees;
            res  = struct('CATE', cate, 'ATE', mean(cate));
        end

        % ---- Doubly-Robust Learner (Kennedy 2023) ----
        function res = estimateDrLearner(obj, X, Y, W)
            N  = size(X, 1);
            pc = obj.PropClip;
            K  = 5;                          % cross-fitting folds
            cv = cvpartition(N, 'KFold', K);

            mu1_hat = zeros(N,1);
            mu0_hat = zeros(N,1);
            e_hat   = zeros(N,1);

            learner = templateTree('MinLeafSize', obj.MinLeafSize, ...
                                   'NumVariablesToSample', obj.Mtry);
            bagArgs = {'Method','Bag','NumLearningCycles',obj.NumTrees,'Learners',learner};

            % --- cross-fitted nuisance estimation ---
            for k = 1:K
                trn = cv.training(k);
                tst = cv.test(k);
                Xt = X(trn,:);  Yt = Y(trn);  Wt = W(trn);

                % mu_1(x) = E[Y|X,W=1]
                m1 = fitrensemble(Xt(Wt==1,:), Yt(Wt==1), bagArgs{:});
                mu1_hat(tst) = predict(m1, X(tst,:));

                % mu_0(x) = E[Y|X,W=0]
                m0 = fitrensemble(Xt(Wt==0,:), Yt(Wt==0), bagArgs{:});
                mu0_hat(tst) = predict(m0, X(tst,:));

                % e(x) = P(W=1|X)  via regression on binary W
                me = fitrensemble(Xt, Wt, bagArgs{:});
                e_hat(tst) = predict(me, X(tst,:));
            end

            % clip propensity
            e_hat = max(min(e_hat, 1-pc), pc);

            % --- AIPW pseudo-outcome ---
            phi = mu1_hat - mu0_hat ...
                + W .* (Y - mu1_hat) ./ e_hat ...
                - (1 - W) .* (Y - mu0_hat) ./ (1 - e_hat);

            % --- second-stage: regress phi on X ---
            finalModel = fitrensemble(X, phi, bagArgs{:});
            cate = predict(finalModel, X);

            res = struct('CATE', cate, 'ATE', mean(cate), ...
                         'PseudoOutcome', phi, ...
                         'Propensity', e_hat);
        end

        % ---- Quantile Treatment Effects (QTE) ----
        function res = estimateQte(obj, X, Y, W)
            %ESTIMATEQTE  Conditional & unconditional quantile treatment effects.
            %   Uses TreeBagger quantile forests (Meinshausen 2006) to estimate
            %   Q_q(Y|X,W=1) and Q_q(Y|X,W=0) for each quantile q, then:
            %     CQTE(q, x) = Q_q(Y|X=x,W=1) - Q_q(Y|X=x,W=0)
            %     QTE(q)     = mean_x CQTE(q, x)

            N  = size(X, 1);
            qs = obj.Quantiles(:)';    % 1 x nQ row vector
            nQ = numel(qs);

            mtry = obj.Mtry;
            if isempty(mtry); mtry = max(1, floor(sqrt(size(X,2)))); end

            % --- Build quantile random forests for treated / control ---
            bag1 = TreeBagger(obj.NumTrees, X(W==1,:), Y(W==1), ...
                'Method', 'regression', ...
                'MinLeafSize', obj.MinLeafSize, ...
                'NumPredictorsToSample', mtry, ...
                'OOBPrediction', 'off');

            bag0 = TreeBagger(obj.NumTrees, X(W==0,:), Y(W==0), ...
                'Method', 'regression', ...
                'MinLeafSize', obj.MinLeafSize, ...
                'NumPredictorsToSample', mtry, ...
                'OOBPrediction', 'off');

            % --- Conditional quantile predictions ---
            Q1 = quantilePredict(bag1, X, 'Quantile', qs);   % N x nQ
            Q0 = quantilePredict(bag0, X, 'Quantile', qs);   % N x nQ

            CQTE = Q1 - Q0;                        % N x nQ  conditional QTE
            QTE  = mean(CQTE, 1);                   % 1 x nQ  unconditional QTE

            % also compute median-based ATE (median quantile) and mean CATE
            [~, midIdx] = min(abs(qs - 0.5));
            cate = CQTE(:, midIdx);                 % N x 1  CATE at median

            res = struct('CQTE',      CQTE, ...
                         'QTE',       QTE, ...
                         'Quantiles', qs, ...
                         'CATE',      cate, ...
                         'ATE',       mean(cate), ...
                         'Q1',        Q1, ...
                         'Q0',        Q0);
        end
    end

    % ================================================================
    %  PRIVATE — causal tree helpers
    % ================================================================
    methods (Access = private)

        function tree = buildCausalTree(obj, X, Y, W, minLeaf, mtry, p, maxDepth) %#ok<INUSL>
            %BUILDCAUSALTREE  Iterative construction of a causal CART tree.
            %   Splitting criterion: maximize  n_L*tau_L^2 + n_R*tau_R^2
            %   (Athey & Imbens 2016, eq. 3).
            %
            %   Tree stored as flat parallel arrays for fast traversal.

            maxNodes = 2^(min(maxDepth,15)+1);
            feat  = zeros(maxNodes, 1);     % 0 = leaf
            thresh = zeros(maxNodes, 1);
            lc    = zeros(maxNodes, 1);     % left child id
            rc    = zeros(maxNodes, 1);     % right child id
            tau   = zeros(maxNodes, 1);     % leaf CATE
            nNodes = 1;

            % stack entries: {nodeId, sampleIdx_vector, depth}
            stack = {1, (1:size(X,1))', 0};

            while ~isempty(stack)
                nodeId  = stack{end,1};
                samples = stack{end,2};
                depth   = stack{end,3};
                stack(end,:) = [];

                Xn = X(samples,:);
                Yn = Y(samples);
                Wn = W(samples);
                n  = numel(samples);
                n1 = sum(Wn == 1);
                n0 = n - n1;

                % --- stopping rules ---
                if n < 2*minLeaf || n1 < 2 || n0 < 2 || depth >= maxDepth
                    if n1 > 0 && n0 > 0
                        tau(nodeId) = mean(Yn(Wn==1)) - mean(Yn(Wn==0));
                    end
                    continue
                end

                % --- find best causal split ---
                [bF, bT, bG] = obj.findBestSplit( ...
                    Xn, Yn, Wn, mtry, p, minLeaf);

                if bG <= 0   % no valid split
                    tau(nodeId) = mean(Yn(Wn==1)) - mean(Yn(Wn==0));
                    continue
                end

                leftMask = Xn(:, bF) <= bT;
                leftId   = nNodes + 1;
                rightId  = nNodes + 2;
                nNodes   = nNodes + 2;

                feat(nodeId)   = bF;
                thresh(nodeId) = bT;
                lc(nodeId)     = leftId;
                rc(nodeId)     = rightId;

                stack = [stack; {leftId,  samples(leftMask),  depth+1}]; %#ok
                stack = [stack; {rightId, samples(~leftMask), depth+1}]; %#ok
            end

            tree.feat   = feat(1:nNodes);
            tree.thresh = thresh(1:nNodes);
            tree.lc     = lc(1:nNodes);
            tree.rc     = rc(1:nNodes);
            tree.tau    = tau(1:nNodes);
            tree.nNodes = nNodes;
        end

        function [bestF, bestT, bestG] = findBestSplit(~, X, Y, W, mtry, p, minLeaf)
            %FINDBESTSPLIT  Causal splitting criterion using cumulative sums.
            %   O(n log n) per candidate feature.

            n = size(X, 1);
            bestG = -Inf;  bestF = 0;  bestT = 0;

            candidates = randsample(p, min(mtry, p), false);

            for jj = 1:numel(candidates)
                j = candidates(jj);
                [xj, si] = sort(X(:,j));
                yj = Y(si);
                wj = W(si);

                w1 = (wj == 1);
                cumN1 = cumsum(w1);
                cumN0 = cumsum(~w1);
                cumY1 = cumsum(yj .* w1);
                cumY0 = cumsum(yj .* (~w1));

                totN1 = cumN1(end);  totN0 = cumN0(end);
                totY1 = cumY1(end);  totY0 = cumY0(end);

                for i = minLeaf:(n - minLeaf)
                    if xj(i) == xj(i+1); continue; end  % skip ties

                    n1L = cumN1(i);  n0L = cumN0(i);
                    n1R = totN1 - n1L;  n0R = totN0 - n0L;

                    if n1L < 1 || n0L < 1 || n1R < 1 || n0R < 1
                        continue
                    end

                    tauL = cumY1(i)/n1L  - cumY0(i)/n0L;
                    tauR = (totY1 - cumY1(i))/n1R - (totY0 - cumY0(i))/n0R;

                    gain = i * tauL^2 + (n - i) * tauR^2;

                    if gain > bestG
                        bestG = gain;
                        bestF = j;
                        bestT = (xj(i) + xj(i+1)) / 2;
                    end
                end
            end
        end

        function tree = honestReestimate(~, tree, X_est, Y_est, W_est)
            %HONESTREESTIMATE  Re-estimate leaf effects with held-out data.

            N = size(X_est, 1);
            leafIds = zeros(N, 1);

            % assign each estimation observation to a leaf
            for i = 1:N
                nid = 1;
                while tree.feat(nid) > 0
                    if X_est(i, tree.feat(nid)) <= tree.thresh(nid)
                        nid = tree.lc(nid);
                    else
                        nid = tree.rc(nid);
                    end
                end
                leafIds(i) = nid;
            end

            % reset all leaf tau
            isLeaf = (tree.feat == 0);
            tree.tau(isLeaf) = 0;

            % honest estimation per leaf
            uLeaves = unique(leafIds);
            for j = 1:numel(uLeaves)
                lid  = uLeaves(j);
                mask = (leafIds == lid);
                Yl   = Y_est(mask);
                Wl   = W_est(mask);
                n1   = sum(Wl == 1);
                n0   = sum(Wl == 0);
                if n1 > 0 && n0 > 0
                    tree.tau(lid) = mean(Yl(Wl==1)) - mean(Yl(Wl==0));
                end
            end
        end

        function tau = predictTree(~, tree, X)
            %PREDICTTREE  Predict CATE from a flat causal tree.

            N   = size(X, 1);
            tau = zeros(N, 1);
            for i = 1:N
                nid = 1;
                while tree.feat(nid) > 0
                    if X(i, tree.feat(nid)) <= tree.thresh(nid)
                        nid = tree.lc(nid);
                    else
                        nid = tree.rc(nid);
                    end
                end
                tau(i) = tree.tau(nid);
            end
        end
    end
end
