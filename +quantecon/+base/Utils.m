classdef Utils
    % UTILS Core utility functions for the quantecon package
    %
    %   Static Methods:
    %       colvec - Ensure input is a column vector
    %       rowvec - Ensure input is a row vector
    %       add_const - Add a constant term (column of ones) to a matrix
    %       lag - Lag a time series matrix

    methods (Static)
        function x = colvec(x)
            % COLVEC Ensure input is a column vector
            arguments
                x (:,:)
            end
            if isrow(x)
                x = x';
            end
        end

        function x = rowvec(x)
            % ROWVEC Ensure input is a row vector
            arguments
                x (:,:)
            end
            if iscolumn(x)
                x = x';
            end
        end

        function X = add_const(X)
            % ADD_CONST Add a constant term (column of ones) to a matrix
            %
            %   Usage:
            %       X_new = quantecon.base.Utils.add_const(X)

            arguments
                X (:,:) double
            end

            [T, ~] = size(X);
            X = [ones(T, 1), X];
        end

        function Y = lag(X, k, pad)
            % LAG Lag a time series matrix
            %
            %   Usage:
            %       Y = quantecon.base.Utils.lag(X, k, pad)
            %
            %   Inputs:
            %       X   - (T x N) Data matrix
            %       k   - Number of lags (default: 1)
            %       pad - Value to pad with (default: NaN)

            arguments
                X (:,:) double
                k (1,1) double {mustBeInteger, mustBeNonnegative} = 1
                pad (1,1) double = NaN
            end

            [T, N] = size(X);
            if k >= T
                Y = repmat(pad, T, N);
            else
                Y = [repmat(pad, k, N); X(1:T-k, :)];
            end
        end
    end
end
