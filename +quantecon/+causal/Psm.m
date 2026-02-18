classdef Psm < handle
    %PSM Propensity Score Matching for Causal Inference
    %   Estimates treatment effects (ATT, ATE, ATNT) via propensity score
    %   matching.  Supports multiple PS models, matching algorithms, balance
    %   diagnostics, Abadie-Imbens standard errors, and PSM-DID integration.
    %
    %   References:
    %     Rosenbaum & Rubin (1983) - Propensity Score Framework
    %     Abadie & Imbens (2006, 2011, 2016) - Matching SE & Bias Correction
    %     Heckman, Ichimura & Todd (1997) - Kernel Matching
    %     Caliendo & Kopeinig (2008) - PSM in Practice
    %
    %   PS Models:
    %     "logit"  - Logistic regression (default)
    %     "probit" - Probit regression (glmfit + normcdf link)
    %     "lasso"  - L1-penalized logistic (lassoglm)
    %     "rf"     - Random forest classifier (TreeBagger)
    %
    %   Matching Methods:
    %     "nearest" - Nearest-neighbor matching (1:k, with/without replacement)
    %     "kernel"  - Epanechnikov / Gaussian kernel matching
    %     "radius"  - Radius (caliper) matching
    %
    %   Usage:
    %     % --- Basic PSM (matches Stata teffects psmatch default) ---
    %     psm = quantecon.causal.Psm('WithReplacement',false,'BiasCorrection',true);
    %     res = psm.estimate(Y, W, X);
    %     fprintf('ATT = %.4f (SE = %.4f)\n', res.ATT, res.SE_ATT);
    %
    %     % --- NN(1:5) with replacement, no bias correction ---
    %     psm = quantecon.causal.Psm('NumNeighbors',5,'BiasCorrection',false);
    %     res = psm.estimate(Y, W, X);
    %
    %     % --- Kernel matching with probit PS ---
    %     psm = quantecon.causal.Psm('PSModel','probit','MatchMethod','kernel');
    %     res = psm.estimate(Y, W, X);
    %
    %     % --- ML propensity score with random forest ---
    %     psm = quantecon.causal.Psm('PSModel','rf','MatchMethod','nearest','NumNeighbors',5);
    %     res = psm.estimate(Y, W, X);
    %
    %     % --- PSM-DID (Heckman, Ichimura & Todd 1997) ---
    %     res = psm.estimateDid(Y_pre, Y_post, W, X);
    %     fprintf('PSM-DID ATT = %.4f (SE = %.4f)\n', res.ATT_DID, res.SE_DID);
    %
    %     % --- Balance diagnostics ---
    %     disp(res.Balance)

    properties
        PSModel (1,1) string {mustBeMember(PSModel, ...
            ["logit","probit","lasso","rf"])} = "logit"
        MatchMethod (1,1) string {mustBeMember(MatchMethod, ...
            ["nearest","kernel","radius"])} = "nearest"
        NumNeighbors (1,1) double {mustBeInteger, mustBePositive} = 1
        WithReplacement (1,1) logical = true
        Caliper (1,1) double {mustBePositive} = Inf   % no caliper by default
        KernelType (1,1) string {mustBeMember(KernelType, ...
            ["epanechnikov","gaussian"])} = "epanechnikov"
        Bandwidth (1,1) double {mustBePositive} = 0.06
        CommonSupport (1,1) logical = true
        NumTrees (1,1) double {mustBeInteger, mustBePositive} = 200  % for rf PS
        NumBootstrap (1,1) double {mustBeInteger, mustBePositive} = 200
        BiasCorrection (1,1) logical = true   % Abadie-Imbens (2011) bias correction
        SEMethod (1,1) string {mustBeMember(SEMethod, ...
            ["ai","bootstrap"])} = "ai"   % Abadie-Imbens or bootstrap
    end

    properties (SetAccess = private)
        Results
        PScore        % estimated propensity scores (N,1)
    end

    methods
        function obj = Psm(varargin)
            if nargin > 0
                obj.set(varargin{:});
            end
        end

        function obj = set(obj, varargin)
            p = inputParser;
            addParameter(p, 'PSModel',        obj.PSModel);
            addParameter(p, 'MatchMethod',     obj.MatchMethod);
            addParameter(p, 'NumNeighbors',    obj.NumNeighbors);
            addParameter(p, 'WithReplacement', obj.WithReplacement);
            addParameter(p, 'Caliper',         obj.Caliper);
            addParameter(p, 'KernelType',      obj.KernelType);
            addParameter(p, 'Bandwidth',       obj.Bandwidth);
            addParameter(p, 'CommonSupport',   obj.CommonSupport);
            addParameter(p, 'NumTrees',        obj.NumTrees);
            addParameter(p, 'NumBootstrap',    obj.NumBootstrap);
            addParameter(p, 'BiasCorrection',  obj.BiasCorrection);
            addParameter(p, 'SEMethod',        obj.SEMethod);
            parse(p, varargin{:});
            flds = fieldnames(p.Results);
            for i = 1:numel(flds)
                obj.(flds{i}) = p.Results.(flds{i});
            end
        end

        function res = estimate(obj, Y, W, X)
            %ESTIMATE  Propensity score matching estimation.
            %   Y : (N,1) outcome variable
            %   W : (N,1) binary treatment indicator (0/1)
            %   X : (N,p) covariates

            Y = Y(:);  W = W(:);
            N = length(Y);
            assert(length(W) == N, 'Psm:DimMismatch', 'Y and W must have same length.');
            assert(size(X,1) == N, 'Psm:DimMismatch', 'X must have N rows.');

            % === Step 1: Estimate propensity score ===
            ps = obj.estimatePS(X, W);
            obj.PScore = ps;

            % === Step 2: Common support trimming ===
            if obj.CommonSupport
                psT = ps(W==1);  psC = ps(W==0);
                lo = max(min(psT), min(psC));
                hi = min(max(psT), max(psC));
                keep = (ps >= lo) & (ps <= hi);
            else
                keep = true(N, 1);
            end

            Yk = Y(keep);  Wk = W(keep);  Xk = X(keep,:);  psk = ps(keep);
            Nk = sum(keep);
            idx1 = find(Wk == 1);   % treated indices (in trimmed sample)
            idx0 = find(Wk == 0);   % control indices
            N1 = numel(idx1);  N0 = numel(idx0);

            % === Step 3: Matching ===
            switch obj.MatchMethod
                case "nearest"
                    [matchIdx, matchW] = obj.matchNearest(psk, idx1, idx0);
                case "kernel"
                    [matchIdx, matchW] = obj.matchKernel(psk, idx1, idx0);
                case "radius"
                    [matchIdx, matchW] = obj.matchRadius(psk, idx1, idx0);
            end

            % === Step 4: Treatment effects (with optional bias correction) ===
            % ATT: for each treated, impute counterfactual from matched controls
            att_i = zeros(N1, 1);
            for i = 1:N1
                mIdx = matchIdx{i};
                mW   = matchW{i};
                if isempty(mIdx)
                    att_i(i) = NaN;
                else
                    att_i(i) = Yk(idx1(i)) - sum(Yk(mIdx) .* mW);
                end
            end

            % Abadie-Imbens (2011) bias correction on covariates
            %   bc = mu_hat(X_focal) - weighted mu_hat(X_matched)
            %   ATT correction SUBTRACTS bc (because bc adjusts the imputed
            %   counterfactual upward, which reduces the treatment effect)
            if obj.BiasCorrection && obj.MatchMethod == "nearest"
                bc_att = obj.biasCorrect(Xk, Yk, Wk, idx1, idx0, matchIdx, matchW, 0);
                att_i = att_i - bc_att;
            end

            validATT = ~isnan(att_i);
            ATT = mean(att_i(validATT));
            N1_eff = sum(validATT);

            % ATNT: match controls to treated (reverse)
            [matchIdxR, matchWR] = obj.matchReverse(psk, idx1, idx0);
            atnt_i = zeros(N0, 1);
            for i = 1:N0
                mIdx = matchIdxR{i};
                mW   = matchWR{i};
                if isempty(mIdx)
                    atnt_i(i) = NaN;
                else
                    atnt_i(i) = sum(Yk(mIdx) .* mW) - Yk(idx0(i));
                end
            end

            if obj.BiasCorrection && obj.MatchMethod == "nearest"
                bc_atnt = obj.biasCorrect(Xk, Yk, Wk, idx0, idx1, matchIdxR, matchWR, 1);
                atnt_i = atnt_i + bc_atnt;
            end

            validATNT = ~isnan(atnt_i);
            ATNT = mean(atnt_i(validATNT));
            N0_eff = sum(validATNT);

            % ATE = weighted combination
            ATE = (N1_eff * ATT + N0_eff * ATNT) / (N1_eff + N0_eff);

            % === Step 5: Standard errors ===
            if obj.SEMethod == "ai"
                SE_ATT  = obj.seAbadieImbens(Yk, Xk, Wk, idx1, idx0, matchIdx, matchW, att_i, 'att');
                SE_ATNT = obj.seAbadieImbens(Yk, Xk, Wk, idx0, idx1, matchIdxR, matchWR, atnt_i, 'atnt');
                SE_ATE  = sqrt((N1_eff/(N1_eff+N0_eff))^2 * SE_ATT^2 + ...
                               (N0_eff/(N1_eff+N0_eff))^2 * SE_ATNT^2);
            else
                [SE_ATT, SE_ATNT, SE_ATE] = obj.seBootstrap(Yk, Wk, Xk, psk);
            end

            % === Step 6: Balance diagnostics ===
            bal = obj.balanceDiagnostics(Xk, Wk, psk, matchIdx, matchW, idx1, idx0);

            % === Assemble results ===
            res = struct();
            res.ATT  = ATT;   res.SE_ATT  = SE_ATT;
            res.ATE  = ATE;   res.SE_ATE  = SE_ATE;
            res.ATNT = ATNT;  res.SE_ATNT = SE_ATNT;
            res.N_Total   = N;
            res.N_Trimmed = Nk;
            res.N_Treated = N1_eff;
            res.N_Control = N0_eff;
            res.PScore    = ps;
            res.CommonSupportRange = [lo, hi];
            res.Balance = bal;
            res.MatchMethod = char(obj.MatchMethod);
            res.PSModel     = char(obj.PSModel);
            res.BiasCorrection = obj.BiasCorrection;

            obj.Results = res;
        end

        function res = estimateDid(obj, Y_pre, Y_post, W, X)
            %ESTIMATEDID  PSM-DID: Propensity Score Matching + Difference-in-Differences
            %   Matches treated and control units on pre-treatment covariates X,
            %   then estimates ATT via DID on matched pairs:
            %     ATT_DID = mean(DeltaY_treated) - weighted_mean(DeltaY_matched_control)
            %   where DeltaY_i = Y_post_i - Y_pre_i
            %
            %   Inputs:
            %     Y_pre  : (N,1) pre-treatment outcome
            %     Y_post : (N,1) post-treatment outcome
            %     W      : (N,1) treatment group indicator (0/1, time-invariant)
            %     X      : (N,p) pre-treatment covariates for matching
            %
            %   Reference:
            %     Heckman, Ichimura & Todd (1997, 1998)

            Y_pre  = Y_pre(:);
            Y_post = Y_post(:);
            W = W(:);
            N = length(Y_pre);
            assert(length(Y_post)==N && length(W)==N && size(X,1)==N, ...
                'Psm:DimMismatch', 'All inputs must have the same number of rows.');

            % Step 1: Estimate PS on pre-treatment covariates
            ps = obj.estimatePS(X, W);
            obj.PScore = ps;

            % Step 2: Common support
            if obj.CommonSupport
                psT = ps(W==1);  psC = ps(W==0);
                lo = max(min(psT), min(psC));
                hi = min(max(psT), max(psC));
                keep = (ps >= lo) & (ps <= hi);
            else
                keep = true(N,1);
                lo = min(ps); hi = max(ps);
            end

            Wk = W(keep);  Xk = X(keep,:);  psk = ps(keep);
            Y_pre_k  = Y_pre(keep);
            Y_post_k = Y_post(keep);
            Nk = sum(keep);

            idx1 = find(Wk == 1);
            idx0 = find(Wk == 0);
            N1 = numel(idx1);

            % Step 3: Match treated to controls
            switch obj.MatchMethod
                case "nearest"
                    [matchIdx, matchW] = obj.matchNearest(psk, idx1, idx0);
                case "kernel"
                    [matchIdx, matchW] = obj.matchKernel(psk, idx1, idx0);
                case "radius"
                    [matchIdx, matchW] = obj.matchRadius(psk, idx1, idx0);
            end

            % Step 4: DID on matched sample
            %   DeltaY_i = Y_post_i - Y_pre_i
            DeltaY = Y_post_k - Y_pre_k;

            att_did_i = zeros(N1, 1);
            valid = false(N1, 1);
            for i = 1:N1
                mi = matchIdx{i};
                mw = matchW{i};
                if isempty(mi); continue; end
                % DID for this treated-control pair
                att_did_i(i) = DeltaY(idx1(i)) - sum(DeltaY(mi) .* mw);
                valid(i) = true;
            end

            ATT_DID = mean(att_did_i(valid));
            N1_eff  = sum(valid);

            % Step 5: Bootstrap SE for DID
            B = obj.NumBootstrap;
            att_b = zeros(B, 1);
            for b = 1:B
                bi1 = idx1(randi(N1, N1, 1));
                bi0 = idx0(randi(numel(idx0), numel(idx0), 1));
                bIdx = [bi1; bi0];
                psb = obj.estimatePS(Xk(bIdx,:), Wk(bIdx));
                idxT = find(Wk(bIdx)==1);
                idxC = find(Wk(bIdx)==0);
                switch obj.MatchMethod
                    case "nearest"
                        [mI, mW_b] = obj.matchNearest(psb, idxT, idxC);
                    case "kernel"
                        [mI, mW_b] = obj.matchKernel(psb, idxT, idxC);
                    case "radius"
                        [mI, mW_b] = obj.matchRadius(psb, idxT, idxC);
                end
                DY_b = Y_post_k(bIdx) - Y_pre_k(bIdx);
                a = zeros(numel(idxT),1);
                for j = 1:numel(idxT)
                    if ~isempty(mI{j})
                        a(j) = DY_b(idxT(j)) - sum(DY_b(mI{j}) .* mW_b{j});
                    else
                        a(j) = NaN;
                    end
                end
                att_b(b) = mean(a, 'omitnan');
            end
            SE_DID = std(att_b, 'omitnan');

            % Step 6: Balance
            bal = obj.balanceDiagnostics(Xk, Wk, psk, matchIdx, matchW, idx1, idx0);

            % Assemble
            res = struct();
            res.ATT_DID   = ATT_DID;
            res.SE_DID    = SE_DID;
            res.tstat_DID = ATT_DID / SE_DID;
            res.pval_DID  = 2*(1 - normcdf(abs(res.tstat_DID)));
            res.N_Total   = N;
            res.N_Trimmed = Nk;
            res.N_Treated = N1_eff;
            res.PScore    = ps;
            res.CommonSupportRange = [lo, hi];
            res.Balance   = bal;

            obj.Results = res;
        end
    end

    % ================================================================
    %  PRIVATE — Propensity Score Estimation
    % ================================================================
    methods (Access = private)

        function ps = estimatePS(obj, X, W)
            %ESTIMATEPS  Estimate propensity score P(W=1|X).

            N = size(X, 1);
            switch obj.PSModel
                case "logit"
                    b = glmfit(X, W, 'binomial', 'link', 'logit');
                    ps = glmval(b, X, 'logit');

                case "probit"
                    b = glmfit(X, W, 'binomial', 'link', 'probit');
                    ps = glmval(b, X, 'probit');

                case "lasso"
                    % L1-penalized logistic regression via lassoglm
                    [B, FitInfo] = lassoglm(X, W, 'binomial', ...
                        'CV', 5, 'Link', 'logit');
                    bestIdx = FitInfo.IndexMinDeviance;
                    bLasso  = B(:, bestIdx);
                    intercept = FitInfo.Intercept(bestIdx);
                    linPred = X * bLasso + intercept;
                    ps = 1 ./ (1 + exp(-linPred));

                case "rf"
                    bag = TreeBagger(obj.NumTrees, X, W, ...
                        'Method', 'classification', ...
                        'OOBPrediction', 'on', ...
                        'MinLeafSize', 5);
                    [~, scores] = predict(bag, X);
                    % scores(:,2) = P(W=1|X)
                    ps = scores(:, 2);
            end

            % clip to avoid exact 0/1
            ps = max(min(ps, 1-1e-6), 1e-6);
        end

        % ================================================================
        %  PRIVATE — Matching Algorithms
        % ================================================================

        function [matchIdx, matchW] = matchNearest(obj, ps, idx1, idx0)
            %MATCHNEAREST  k-nearest-neighbor matching (treated -> control).
            %   Without replacement: greedy algorithm, treated units sorted by
            %   minimum distance to any control (hardest-to-match first) to
            %   improve overall matching quality (similar to Stata teffects).
            N1 = numel(idx1);
            matchIdx = cell(N1, 1);
            matchW   = cell(N1, 1);

            if ~obj.WithReplacement
                % Greedy without-replacement: process in data order
                % (consistent with Stata teffects psmatch default behaviour)
                used = false(numel(idx0), 1);

                for i = 1:N1
                    dist = abs(ps(idx1(i)) - ps(idx0));

                    % Caliper + availability filter
                    available = (dist <= obj.Caliper) & ~used;
                    if ~any(available)
                        matchIdx{i} = [];  matchW{i} = [];
                        continue;
                    end

                    distAvail = dist;
                    distAvail(~available) = Inf;
                    [~, sortI] = sort(distAvail);
                    k = min(obj.NumNeighbors, sum(available));
                    mi = sortI(1:k);

                    matchIdx{i} = idx0(mi);
                    matchW{i}   = ones(k,1) / k;
                    used(mi) = true;
                end
            else
                % With replacement (standard)
                for i = 1:N1
                    dist = abs(ps(idx1(i)) - ps(idx0));
                    withinCaliper = dist <= obj.Caliper;
                    if ~any(withinCaliper)
                        matchIdx{i} = [];  matchW{i} = [];
                        continue;
                    end
                    distF = dist;
                    distF(~withinCaliper) = Inf;
                    [~, sortI] = sort(distF);
                    k = min(obj.NumNeighbors, sum(withinCaliper));
                    mi = sortI(1:k);
                    matchIdx{i} = idx0(mi);
                    matchW{i}   = ones(k,1) / k;
                end
            end
        end

        function [matchIdx, matchW] = matchKernel(obj, ps, idx1, idx0)
            %MATCHKERNEL  Kernel matching (treated -> control).
            N1 = numel(idx1);
            h  = obj.Bandwidth;
            matchIdx = cell(N1, 1);
            matchW   = cell(N1, 1);

            for i = 1:N1
                u = (ps(idx0) - ps(idx1(i))) / h;
                switch obj.KernelType
                    case "epanechnikov"
                        kw = max(0, 0.75 * (1 - u.^2));
                    case "gaussian"
                        kw = normpdf(u);
                end
                sumK = sum(kw);
                if sumK > 0
                    matchIdx{i} = idx0;
                    matchW{i}   = kw / sumK;
                else
                    matchIdx{i} = [];
                    matchW{i}   = [];
                end
            end
        end

        function [matchIdx, matchW] = matchRadius(obj, ps, idx1, idx0)
            %MATCHRADIUS  Radius (caliper) matching.
            N1 = numel(idx1);
            cal = obj.Caliper;
            matchIdx = cell(N1, 1);
            matchW   = cell(N1, 1);

            for i = 1:N1
                dist = abs(ps(idx1(i)) - ps(idx0));
                within = dist <= cal;
                if any(within)
                    mi = find(within);
                    matchIdx{i} = idx0(mi);
                    matchW{i}   = ones(numel(mi), 1) / numel(mi);
                else
                    matchIdx{i} = [];
                    matchW{i}   = [];
                end
            end
        end

        function [matchIdx, matchW] = matchReverse(obj, ps, idx1, idx0)
            %MATCHREVERSE  Match controls to nearest treated (for ATNT).
            N0 = numel(idx0);
            matchIdx = cell(N0, 1);
            matchW   = cell(N0, 1);

            switch obj.MatchMethod
                case "nearest"
                    for i = 1:N0
                        dist = abs(ps(idx0(i)) - ps(idx1));
                        withinCaliper = dist <= obj.Caliper;
                        if ~any(withinCaliper)
                            matchIdx{i} = [];  matchW{i} = [];
                            continue;
                        end
                        distF = dist;
                        distF(~withinCaliper) = Inf;
                        [~, sortI] = sort(distF);
                        k = min(obj.NumNeighbors, sum(withinCaliper));
                        mi = sortI(1:k);
                        matchIdx{i} = idx1(mi);
                        matchW{i}   = ones(k,1) / k;
                    end

                case "kernel"
                    h = obj.Bandwidth;
                    for i = 1:N0
                        u = (ps(idx1) - ps(idx0(i))) / h;
                        switch obj.KernelType
                            case "epanechnikov"
                                kw = max(0, 0.75 * (1 - u.^2));
                            case "gaussian"
                                kw = normpdf(u);
                        end
                        sumK = sum(kw);
                        if sumK > 0
                            matchIdx{i} = idx1;
                            matchW{i}   = kw / sumK;
                        else
                            matchIdx{i} = [];  matchW{i} = [];
                        end
                    end

                case "radius"
                    cal = obj.Caliper;
                    for i = 1:N0
                        dist = abs(ps(idx0(i)) - ps(idx1));
                        within = dist <= cal;
                        if any(within)
                            mi = find(within);
                            matchIdx{i} = idx1(mi);
                            matchW{i}   = ones(numel(mi),1) / numel(mi);
                        else
                            matchIdx{i} = [];  matchW{i} = [];
                        end
                    end
            end
        end

        % ================================================================
        %  PRIVATE — Standard Errors
        % ================================================================

        function se = seAbadieImbens(~, Y, X, W, idxFocal, idxDonor, matchIdx, matchW, te_i, type) %#ok<INUSL>
            %SEABADIEIMBENS  Abadie & Imbens (2006, 2016) heteroskedasticity-
            %   consistent variance estimator for matching.
            %
            %   V_AI = (1/Nf^2) * sum_i [ sigma^2_f(Xi) + K_M(i)^2 * sigma^2_d(Xi)/M^2 ]
            %
            %   sigma^2_w(Xi) estimated from J closest same-group neighbors (J=min(M,3)).
            %   K_M(i) = number of times donor unit i is used as a match.

            Nf = numel(idxFocal);
            Nd = numel(idxDonor);

            valid = ~isnan(te_i);
            Nv = sum(valid);
            if Nv < 2; se = NaN; return; end

            % --- Compute K_M(i): match frequency for each donor ---
            KM = zeros(Nd, 1);
            for i = 1:Nf
                mi = matchIdx{i};
                if isempty(mi); continue; end
                for j = 1:numel(mi)
                    loc = find(idxDonor == mi(j), 1);
                    if ~isempty(loc)
                        KM(loc) = KM(loc) + 1;
                    end
                end
            end

            % --- Estimate conditional variance via nearest same-group neighbors ---
            J = 3;  % number of neighbors for variance estimation

            % Focal group variance: sigma^2_f(Xi)
            psFocal = zeros(Nf, 1);
            % We don't have PS in this function, use X-based distance for variance
            % But simpler: use matching residuals
            % Alternative: nearest-neighbor variance estimator (AI 2006, eq 12)

            % For focal units: AI (2006 eq.14) — include unit i itself + J nn
            % sigma^2(Xi) = J/(J+1) * s^2_{J,i} = var([Yi; Y_nn], 1)
            Yf = Y(idxFocal);
            Xf = X(idxFocal, :);
            sigma2_f = zeros(Nf, 1);
            for i = 1:Nf
                if ~valid(i); continue; end
                dists = sum((Xf - Xf(i,:)).^2, 2);
                dists(i) = Inf;  % exclude self from NN search
                [~, si] = sort(dists);
                jj = min(J, Nf - 1);
                nn = si(1:jj);
                sigma2_f(i) = var([Yf(i); Yf(nn)], 1);  % pop var over J+1 units
            end

            % For donor units: same AI (2006) formula
            Yd = Y(idxDonor);
            Xd = X(idxDonor, :);
            sigma2_d = zeros(Nd, 1);
            for i = 1:Nd
                if KM(i) == 0; continue; end  % not used as match
                dists = sum((Xd - Xd(i,:)).^2, 2);
                dists(i) = Inf;
                [~, si] = sort(dists);
                jj = min(J, Nd - 1);
                nn = si(1:jj);
                sigma2_d(i) = var([Yd(i); Yd(nn)], 1);  % pop var over J+1 units
            end

            % --- AI (2006) variance formula ---
            % V = (1/Nf^2) * [ sum_focal sigma2_f(i)  +  sum_donor (KM(i)/M)^2 * sigma2_d(i) ]
            M = max(1, numel(matchW{find(valid, 1)}));  % NumNeighbors used

            V = (1 / Nv^2) * ( sum(sigma2_f(valid)) + ...
                                sum((KM / M).^2 .* sigma2_d) );
            se = sqrt(max(V, 0));
        end

        function [se_att, se_atnt, se_ate] = seBootstrap(obj, Y, W, X, ps) %#ok<INUSD>
            %SEBOOTSTRAP  Nonparametric bootstrap standard errors.

            N  = length(Y);
            B  = obj.NumBootstrap;
            att_b  = zeros(B, 1);
            atnt_b = zeros(B, 1);
            ate_b  = zeros(B, 1);

            idx1_full = find(W == 1);
            idx0_full = find(W == 0);

            for b = 1:B
                % Stratified bootstrap (resample treated & control separately)
                bi1 = idx1_full(randi(numel(idx1_full), numel(idx1_full), 1));
                bi0 = idx0_full(randi(numel(idx0_full), numel(idx0_full), 1));
                bIdx = [bi1; bi0];
                Yb = Y(bIdx);  Wb = W(bIdx);  Xb = X(bIdx,:);

                psb = obj.estimatePS(Xb, Wb);
                idxT = find(Wb == 1);
                idxC = find(Wb == 0);

                switch obj.MatchMethod
                    case "nearest"
                        [mI, mW] = obj.matchNearest(psb, idxT, idxC);
                    case "kernel"
                        [mI, mW] = obj.matchKernel(psb, idxT, idxC);
                    case "radius"
                        [mI, mW] = obj.matchRadius(psb, idxT, idxC);
                end

                a = zeros(numel(idxT), 1);
                for i = 1:numel(idxT)
                    if ~isempty(mI{i})
                        a(i) = Yb(idxT(i)) - sum(Yb(mI{i}) .* mW{i});
                    else
                        a(i) = NaN;
                    end
                end
                att_b(b)  = mean(a, 'omitnan');

                % reverse matching for ATNT
                [mIR, mWR] = obj.matchReverse(psb, idxT, idxC);
                c = zeros(numel(idxC), 1);
                for i = 1:numel(idxC)
                    if ~isempty(mIR{i})
                        c(i) = sum(Yb(mIR{i}) .* mWR{i}) - Yb(idxC(i));
                    else
                        c(i) = NaN;
                    end
                end
                atnt_b(b) = mean(c, 'omitnan');
                n1 = numel(idxT); n0 = numel(idxC);
                ate_b(b) = (n1*att_b(b) + n0*atnt_b(b)) / (n1+n0);
            end

            se_att  = std(att_b, 'omitnan');
            se_atnt = std(atnt_b, 'omitnan');
            se_ate  = std(ate_b, 'omitnan');
        end

        % ================================================================
        %  PRIVATE — Abadie-Imbens (2011) Bias Correction
        % ================================================================

        function bc = biasCorrect(~, X, Y, W, idxFocal, idxDonor, matchIdx, matchW, donorW)
            %BIASCORRECT  Abadie & Imbens (2011) regression-based bias correction.
            %
            %   For each focal unit i, the bias correction is:
            %     bc_i = mu_hat(X_focal_i) - sum_j w_j * mu_hat(X_donor_j)
            %   where mu_hat(x) = linear regression of Y on X among donor group.
            %
            %   donorW: treatment indicator value for the donor group (0 or 1)

            Nf = numel(idxFocal);
            bc = zeros(Nf, 1);

            % Build regression model mu(x) = E[Y|X, W=donorW] using all donors
            Xd = X(idxDonor, :);
            Yd = Y(idxDonor);

            % OLS: Y_d = [1, X_d] * beta
            Xd_c = [ones(numel(idxDonor), 1), Xd];
            beta = (Xd_c' * Xd_c) \ (Xd_c' * Yd);

            % Predict mu(x) for focal and donor observations
            for i = 1:Nf
                mi = matchIdx{i};
                mw = matchW{i};
                if isempty(mi)
                    bc(i) = NaN;
                    continue;
                end

                % mu_hat at focal unit's X
                xf = [1, X(idxFocal(i), :)];
                mu_focal = xf * beta;

                % weighted mean of mu_hat at matched donor X
                Xm = [ones(numel(mi),1), X(mi, :)];
                mu_donors = Xm * beta;
                mu_match = sum(mu_donors .* mw);

                % Correction: for ATT (donorW=0), the correction reduces bias
                % by adjusting the imputed counterfactual upward/downward
                bc(i) = mu_focal - mu_match;
            end
        end

        % ================================================================
        %  PRIVATE — Balance Diagnostics
        % ================================================================

        function bal = balanceDiagnostics(~, X, W, ps, matchIdx, matchW, idx1, idx0)
            %BALANCEDIAGNOSTICS  Standardized mean difference before/after matching.
            %   Reports: VarName, MeanT, MeanC_raw, MeanC_matched, SMD_before, SMD_after
            %   SMD = (mean_T - mean_C) / sqrt((var_T + var_C)/2)

            [~, p] = size(X);
            N1 = numel(idx1);

            % Raw means & variances
            meanT = mean(X(idx1,:), 1);   % 1 x p
            meanC = mean(X(idx0,:), 1);
            varT  = var(X(idx1,:), 0, 1);
            varC  = var(X(idx0,:), 0, 1);
            poolSD = sqrt((varT + varC) / 2);
            poolSD(poolSD < 1e-10) = 1;   % avoid div by 0

            smd_before = (meanT - meanC) ./ poolSD;

            % Matched control means (weighted)
            matchedMeanC = zeros(1, p);
            totalW = 0;
            for i = 1:N1
                mi = matchIdx{i};
                mw = matchW{i};
                if ~isempty(mi)
                    matchedMeanC = matchedMeanC + sum(X(mi,:) .* mw, 1);
                    totalW = totalW + 1;
                end
            end
            if totalW > 0
                matchedMeanC = matchedMeanC / totalW;
            end

            smd_after = (meanT - matchedMeanC) ./ poolSD;

            % Variance ratio after matching
            varRatio = varT ./ max(varC, 1e-10);

            % Build table
            varNames = cell(p, 1);
            for j = 1:p
                varNames{j} = sprintf('X%d', j);
            end

            bal = struct();
            bal.VarNames   = varNames;
            bal.MeanTreated  = meanT(:);
            bal.MeanControlRaw     = meanC(:);
            bal.MeanControlMatched = matchedMeanC(:);
            bal.SMD_Before = smd_before(:);
            bal.SMD_After  = smd_after(:);
            bal.VarRatio   = varRatio(:);

            % Summary
            bal.MeanAbsSMD_Before = mean(abs(smd_before));
            bal.MeanAbsSMD_After  = mean(abs(smd_after));
            bal.MaxAbsSMD_After   = max(abs(smd_after));
        end

    end
end
