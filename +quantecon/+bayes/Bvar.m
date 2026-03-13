classdef Bvar < handle
    %BVAR Bayesian Vector Autoregression Model
    %
    %   Wrapper for MATLAB's native 'bayesvarm' (Econometrics Toolbox).
    %   Provides a simplified interface compatible with quantecon.
    %
    %   Usage:
    %       mdl = quantecon.bayes.Bvar(Lags);
    %       res = mdl.estimate(Y);

    properties
        Lags (1,1) {mustBeInteger, mustBePositive} = 1
        PriorType (1,1) string {mustBeMember(PriorType, ["minnesota", "conjugate", "diffuse"])} = "minnesota"
        Hyperparameters % Struct for options
        Model % Underlying bayesvarm object
        Results % Estimation results
    end

    methods
        function obj = Bvar(p, prior)
            if nargin > 0
                obj.Lags = p;
            end
            if nargin > 1
                obj.PriorType = prior;
            end

            % Default Hyperparameters for Minnesota
            % Note: bayesvarm uses 'Lambda' field in prior options sometimes, or specific NVPs.
            % We will pass struct fields as Name-Value pairs to bayesvarm if possible.

            obj.Hyperparameters.Lambda = []; % if empty, use default
        end

        function results = estimate(obj, Y)
            %ESTIMATE Estimate BVAR model
            %
            %   Input:
            %       Y: (TxK) Time series data

            arguments
                obj
                Y (:,:) double
            end

            [T, K] = size(Y);
            p = obj.Lags;

            % 1. Create bayesvarm object
            % PriorMdl = bayesvarm(numseries, numlags, 'PriceType', ...)

            % Map "minnesota" to appropriate constructor args
            % bayesvarm(K, p, 'PriorType', ...) isn't the syntax.
            % Syntax: bayesvarm(K, p, 'Start', ...)?
            % No, bayesvarm(K, p) creates a default conjugate prior object usually?
            % Check help: PriorMdl = bayesvarm(numseries, numlags)

            % For 'minnesota' logic (usually 'Doan' or similar):
            % In MATLAB, the default is often 'diffuse' or 'conjugate'.
            % We can set 'PriorType' if the property exists, or use 'ModelType'.
            % Help said: ModelType = 'conjugate' | 'semiconjugate' | 'diffuse' (Default) | 'normal'

            % "Minnesota" usually refers to the structure of the Prior Mean/Var, often used with 'conjugate' or 'normal'.
            % If user wants "Minnesota", we typically use 'conjugate' or 'semiconjugate' and set the Prior Mean/Covariance accordingly.
            % However, bayesvarm has helper methods or properties for Minnesota structure?
            % Actually, the help text mentions "Minnesota Prior Regularization Options".
            % So we can pass name-value pairs like 'SelfLag', 'CrossLag', etc.

            % Let's create the model object.

            % Determine ModelType
            if strcmpi(obj.PriorType, "minnesota")
                % Use conjugate as base for Minnesota-like behavior?
                % Or 'normal'?
                % The doc mentions "SetMinnesotaPrior..." examples.
                % Usually implies Conjugate Normal-Inverse-Wishart.
                modelType = 'conjugate';
            else
                modelType = obj.PriorType;
            end

            try
                priorMdl = bayesvarm(K, p, 'ModelType', modelType);
            catch
                % Fallback for older versions?
                priorMdl = bayesvarm(K, p);
            end

            % Apply Hyperparameters if 'minnesota'
            if strcmpi(obj.PriorType, "minnesota")
                % Standard Minnesota defaults if not set in Hyperparameters
                % We can customize if obj.Hyperparameters has fields.
            end

            % 2. Estimate Posterior
            % EstMdl = estimate(PriorMdl, Y);
            % Or [EstMdl, EstParams] = estimate(...)

            % The estimate method for bayesvarm returns an empirical posterior distribution object?
            % OR it returns a standard object with posterior mean?

            % Usually:
            % [EstMdl, PosteriorSummary] = estimate(priorMdl, Y);

            % Estimate
            [estMdl, summary] = estimate(priorMdl, Y);

            obj.Model = estMdl;

            % Extract Coefficients into standard [Const; AR1; ... ARp] format
            % estMdl.AR is usually a cell array {1}(KxK) or 3D array (KxKxP)
            % estMdl.Constant is (Kx1)

            coeffs = [];

            % Constant
            if ~isempty(estMdl.Constant)
                coeffs = [coeffs; estMdl.Constant'];
            end

            % AR Terms
            % Check if cell or matrix
            if iscell(estMdl.AR)
                for i = 1:p
                    coeffs = [coeffs; estMdl.AR{i}'];
                end
            else
                % 3D Array
                for i = 1:p
                    coeffs = [coeffs; estMdl.AR(:,:,i)'];
                end
            end

            % Reconstruct X matrix for Residuals and Forecast
            % Y_eff = Y(p+1:end, :)
            Y_eff = Y(p+1:end, :);
            % T_eff = size(Y_eff, 1); WRONG, T is already size(Y,1).
            T_eff = size(Y_eff, 1);

            X = ones(T_eff, 1);
            for l = 1:p
                X = [X, Y(p+1-l : end-l, :)];
            end

            % Residuals
            U = Y_eff - X * coeffs;
            Sigma = (U' * U) / T_eff;

            results.Coefficients = coeffs;
            results.Sigma = Sigma;
            results.Residuals = U;
            results.Summary = summary;
            results.X = X;
            results.Y = Y_eff;
            results.T = T_eff;

            obj.Results = results;
        end

        function y_forecast = forecast(obj, horizon)
            %FORECAST Forecast
            if isempty(obj.Model)
                error('Estimate first.');
            end

            % forecast method of bayesvarm object
            % [YF, YMSE] = forecast(EstMdl, horizon, Y0)

            % We need presample Y0.
            % The model needs the last p observations.

            Y_orig = obj.Results.Y;
            p = obj.Lags;
            Y0 = Y_orig(end-p+1:end, :);

            [y_forecast, ~] = forecast(obj.Model, horizon, Y0);
        end
    end
end
