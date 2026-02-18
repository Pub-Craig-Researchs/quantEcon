classdef Rdd < handle
    %RDD Regression Discontinuity Design Estimator
    %   Estimates Sharp and Fuzzy RDD using Local Polynomial Regression.
    %   Supports automatic bandwidth selection (IK, CCT) and various kernels.

    properties
        Cutoff (1,1) double = 0
        Order (1,1) double {mustBeInteger, mustBeNonnegative} = 1
        Kernel (1,1) string {mustBeMember(Kernel, ["triangular", "uniform", "epanechnikov"])} = "triangular"
        BandwidthMethod (1,1) string {mustBeMember(BandwidthMethod, ["ik", "cct", "rot", "manual"])} = "ik"
        ManualBandwidth (1,1) double {mustBePositive} = 1.0
        Cluster (:,1) double = []
    end

    properties (SetAccess = private)
        Model
        Results
        Bandwidth
    end

    methods
        function obj = Rdd(varargin)
            % Constructor options
            if nargin > 0
                obj.set(varargin{:});
            end
        end

        function obj = set(obj, varargin)
            p = inputParser;
            addParameter(p, 'Cutoff', obj.Cutoff);
            addParameter(p, 'Order', obj.Order);
            addParameter(p, 'Kernel', obj.Kernel);
            addParameter(p, 'BandwidthMethod', obj.BandwidthMethod);
            addParameter(p, 'ManualBandwidth', obj.ManualBandwidth);
            addParameter(p, 'Cluster', obj.Cluster);
            parse(p, varargin{:});

            obj.Cutoff = p.Results.Cutoff;
            obj.Order = p.Results.Order;
            obj.Kernel = p.Results.Kernel;
            obj.BandwidthMethod = p.Results.BandwidthMethod;
            obj.ManualBandwidth = p.Results.ManualBandwidth;
            obj.Cluster = p.Results.Cluster;
        end

        function res = estimate(obj, runVar, outcome, treatment)
            %ESTIMATE Estimate RDD model
            %   res = estimate(obj, runVar, outcome) -> Sharp RDD
            %   res = estimate(obj, runVar, outcome, treatment) -> Fuzzy RDD

            arguments
                obj
                runVar (:,1) double
                outcome (:,1) double
                treatment (:,1) double = []
            end

            % Normalize running variable
            X = runVar - obj.Cutoff;
            Y = outcome;
            N = length(Y);

            % Determine type
            if isempty(treatment)
                isFuzzy = false;
                D = double(X >= 0);
            else
                isFuzzy = true;
                D = treatment;
            end

            % Bandwidth Selection
            if obj.BandwidthMethod == "manual"
                h = obj.ManualBandwidth;
            elseif obj.BandwidthMethod == "ik"
                h = obj.selectBandwidthIK(X, Y, obj.Order);
            elseif obj.BandwidthMethod == "cct"
                h = obj.selectBandwidthCCT(X, Y);
            else % rot
                h = 1.06 * std(X) * N^(-1/5);
            end
            obj.Bandwidth = h;

            % Kernel Weights
            W = obj.computeKernelWeights(X, h, obj.Kernel);

            % Local Polynomial Regression

            % Setup design matrices for Local Poly
            % We fit separate polynomials on left (X<0) and right (X>=0)

            leftIdx = (X < 0) & (X >= -h);
            rightIdx = (X >= 0) & (X <= h);

            res = struct();
            res.Bandwidth = h;
            res.N_Left = sum(leftIdx);
            res.N_Right = sum(rightIdx);

            % Check observations
            if res.N_Left < obj.Order + 2 || res.N_Right < obj.Order + 2
                warning("Insufficient observations within bandwidth %.4f.", h);
            end

            % Estimate Intercepts (limit at X=0)
            [muLeft, varMuLeft] = obj.localPolyLikelihood(X(leftIdx), Y(leftIdx), W(leftIdx), obj.Order);
            [muRight, varMuRight] = obj.localPolyLikelihood(X(rightIdx), Y(rightIdx), W(rightIdx), obj.Order);

            res.Mu_Left = muLeft;
            res.Mu_Right = muRight;
            tauSharp = muRight - muLeft;
            varTauSharp = varMuLeft + varMuRight; % Assuming independence between sides

            if isFuzzy
                % Estimate discontinuity in treatment probability
                [piLeft, ~] = obj.localPolyLikelihood(X(leftIdx), D(leftIdx), W(leftIdx), obj.Order);
                [piRight, ~] = obj.localPolyLikelihood(X(rightIdx), D(rightIdx), W(rightIdx), obj.Order);

                firstStage = piRight - piLeft;
                if abs(firstStage) < 1e-3
                    warning('Weak first stage in Fuzzy RDD.');
                end

                tau = tauSharp / firstStage;
                % Delta method for variance (approximate)
                % var(tau) approx (1/FS^2) * var(Num) + (Num^2/FS^4) * var(FS)
                % Ignoring covariance for simplicity in this port
                varTau = varTauSharp / (firstStage^2);

                res.FirstStage = firstStage;
                res.ReducedForm = tauSharp;
            else
                tau = tauSharp;
                varTau = varTauSharp;
            end

            res.Coefficients = tau;
            res.StandardError = sqrt(varTau);
            res.tStat = tau / res.StandardError;
            res.pValue = 2 * (1 - normcdf(abs(res.tStat)));

            obj.Results = res;
        end
    end

    methods (Access = private)
        function [mu, varMu] = localPolyLikelihood(~, x, y, w, order)
            % Weighted Least Squares for Local Polynomial
            % Returns intercept (predicted at x=0) and its variance

            n = length(x);
            XPoly = ones(n, order + 1);
            for p = 1:order
                XPoly(:, p+1) = x.^p;
            end

            W = diag(w);

            % Beta = (X'WX)^-1 X'Wy
            XtW = XPoly' * W;
            XtWX = XtW * XPoly;

            % Robust check for singularity
            if rcond(XtWX) < 1e-12
                beta = pinv(XtWX) * (XtW * y);
            else
                beta = XtWX \ (XtW * y);
            end

            mu = beta(1);

            % Variance: (X'WX)^-1 (X' W^2 e^2 X) (X'WX)^-1  (HC1 style)
            residuals = y - XPoly * beta;
            We2 = diag(w.^2 .* residuals.^2);

            Meat = XPoly' * We2 * XPoly;
            Inv = pinv(XtWX);

            V = Inv * Meat * Inv;
            varMu = V(1,1);
        end

        function W = computeKernelWeights(~, X, h, kernel)
            u = X / h;
            switch kernel
                case "triangular"
                    W = max(0, 1 - abs(u));
                case "uniform"
                    W = double(abs(u) <= 1);
                case "epanechnikov"
                    W = max(0, 0.75 * (1 - u.^2));
            end
        end

        function h = selectBandwidthIK(~, X, Y, order)
            % Imbens-Kalyanaraman (2012)
            N = length(X);
            % Pilot bandwidth
            hPilot = 1.84 * std(X) * N^(-1/5);

            % Simple quadratic fit to estimate curvature m''(x)
            leftIdx = (X < 0) & (X >= -hPilot);
            rightIdx = (X >= 0) & (X <= hPilot);

            if sum(leftIdx) < 10 || sum(rightIdx) < 10
                h = hPilot; return;
            end

            % Estimate m2 (curvature)
            % Fit quadratic: y = a + bx + cx^2
            pL = polyfit(X(leftIdx), Y(leftIdx), 2);
            pR = polyfit(X(rightIdx), Y(rightIdx), 2);

            m2Left = 2 * pL(1);
            m2Right = 2 * pR(1);

            % Estimate variance
            resL = Y(leftIdx) - polyval(pL, X(leftIdx));
            resR = Y(rightIdx) - polyval(pR, X(rightIdx));
            s2L = var(resL);
            s2R = var(resR);

            % IK Formula (simplified Ck for triangular)
            Ck = 3.4375;
            reg = max(abs(m2Left - m2Right), 0.01 * std(Y)); % Reg to avoid div by zero
            h = Ck * ((s2L + s2R) / (reg^2 * N))^(1/5);
        end

        function h = selectBandwidthCCT(~, X, ~)
            % Calonico-Cattaneo-Titiunik (Simple Proxy)
            N = length(X);
            h = 1.5 * std(X) * N^(-1/5);
        end
    end
end
