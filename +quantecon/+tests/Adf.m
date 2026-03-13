function tstat = Adf(y, IC, maxlag)
% Adf  Augmented Dickey-Fuller test with information-criterion lag selection.
%
%   tstat = quantecon.tests.Adf(y, IC, maxlag)
%
%   Regression:
%       dy_t = alpha + beta * y_{t-1} + sum_{j=1}^{p} gamma_j * dy_{t-j} + e_t
%
%   Returns the t-statistic on beta (right-tailed for bubble detection).
%
%   Inputs:
%       y      - (T0 x 1) time series vector.
%       IC     - Lag selection rule: 0 = fixed lag, 1 = AIC, 2 = BIC.
%       maxlag - Lag order when IC=0; maximum lag order when IC > 0.
%
%   Output:
%       tstat  - ADF t-statistic on the coefficient of y_{t-1}.
%
%   Reference:
%       Phillips, S.-P., Shi, S., Yu, J. (2015). Testing for Multiple
%       Bubbles: Historical Episodes of Exuberance and Collapse in the
%       S&P 500. International Economic Review, 56(4), 1043-1078.
%
%   Original code: Shuping Shi, Macquarie University.
%   [FIX]: Corrected degrees-of-freedom computation for k > 0.
%   [FIX]: Replaced inv(X'X) with backslash for numerical stability.

    arguments
        y (:,1) double
        IC (1,1) double {mustBeMember(IC, [0, 1, 2])} = 0
        maxlag (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    end

    T0 = length(y);
    T1 = T0 - 1;
    dy = y(2:T0) - y(1:T1);
    ylag = y(1:T1);
    const = ones(T1, 1);
    X1 = [ylag, const];                   % base design: [y_{t-1}, 1]

    T = T1 - maxlag;                       % effective sample length

    if IC > 0
        % --- Information-criterion lag selection ---
        nLags = maxlag + 1;
        ICC = zeros(nLags, 1);
        ADF = zeros(nLags, 1);

        for k = 0:maxlag
            [X2, dy01] = buildDesign(X1, dy, maxlag, T1, T, k);

            beta = X2 \ dy01;
            eps = dy01 - X2*beta;

            % [FIX]: correct dof = T - (2 + k), not T - 2
            dof = T - (2 + k);
            npdf = sum(-0.5*log(2*pi) - 0.5*(eps.^2));

            if IC == 1           % AIC
                ICC(k+1) = -2*npdf/T + 2*size(beta, 1)/T;
            else                 % BIC
                ICC(k+1) = -2*npdf/T + size(beta, 1)*log(T)/T;
            end

            % [FIX]: use backslash instead of inv
            sig2 = (eps'*eps) / dof;
            varBeta = sig2 * ((X2'*X2) \ eye(size(X2, 2)));
            ADF(k+1) = beta(1) / sqrt(varBeta(1, 1));
        end

        [~, bestIdx] = min(ICC);
        tstat = ADF(bestIdx);

    else
        % --- Fixed lag order ---
        k = maxlag;
        [X2, dy01] = buildDesign(X1, dy, maxlag, T1, T, k);

        beta = X2 \ dy01;
        eps = dy01 - X2*beta;

        % [FIX]: correct dof
        dof = T - (2 + k);
        sig2 = (eps'*eps) / dof;
        varBeta = sig2 * ((X2'*X2) \ eye(size(X2, 2)));
        tstat = beta(1) / sqrt(varBeta(1, 1));
    end
end

%% ---- Local function --------------------------------------------------------
function [X2, dy01] = buildDesign(X1, dy, maxlag, T1, T, k)
% buildDesign  Construct the ADF regression matrix for lag order k.
%   Rows are always aligned from (maxlag+1) to T1 so that all models
%   share the same effective sample, ensuring comparable IC values.

    xx = X1(maxlag+1:T1, :);              % [y_{t-1}, const] trimmed
    dy01 = dy(maxlag+1:T1);               % dependent variable trimmed

    if k > 0
        X2 = [xx, zeros(T, k)];
        for j = 1:k
            X2(:, size(xx, 2) + j) = dy(maxlag + 1 - j : T1 - j);
        end
    else
        X2 = xx;
    end
end
