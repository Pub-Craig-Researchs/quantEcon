function [beta, eps, lag] = AdfRes(y, IC, maxlag)
% AdfRes  Restricted ADF regression under the unit-root null.
%
%   [beta, eps, lag] = quantecon.tests.AdfRes(y, IC, maxlag)
%
%   Fits the restricted model (no y_{t-1} term):
%       dy_t = mu + sum_{j=1}^{p} gamma_j * dy_{t-j} + e_t
%
%   Used to obtain null-hypothesis residuals for the wild bootstrap
%   procedure (wmboot / PsyBoot).
%
%   Inputs:
%       y      - (T0 x 1) time series vector.
%       IC     - Lag selection: 0 = fixed, 1 = AIC, 2 = BIC.
%       maxlag - Fixed lag (IC=0) or max lag (IC>0).
%
%   Outputs:
%       beta   - ((1+lag) x 1) estimated coefficients [mu; gamma_1; ...].
%       eps    - (T x 1) residual vector (effective sample).
%       lag    - Selected lag order.
%
%   Reference:
%       Phillips, Shi, Yu (2015). Testing for Multiple Bubbles.
%
%   Original code: Shuping Shi, Macquarie University.
%   [FIX]: Replaced inv() with backslash.

    arguments
        y (:,1) double
        IC (1,1) double {mustBeMember(IC, [0, 1, 2])} = 0
        maxlag (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    end

    T0 = length(y);
    T1 = T0 - 1;
    dy = y(2:T0) - y(1:T1);
    const = ones(T1, 1);

    T = T1 - maxlag;

    if IC > 0
        % --- IC-based lag selection ---
        nLags = maxlag + 1;
        ICC = zeros(nLags, 1);
        betaCell = cell(nLags, 1);
        epsCell = cell(nLags, 1);

        for k = 0:maxlag
            [X2, dy01] = buildDesignRes(const, dy, maxlag, T1, T, k);

            b = X2 \ dy01;
            e = dy01 - X2*b;

            betaCell{k+1} = b;
            epsCell{k+1} = e;

            npdf = sum(-0.5*log(2*pi) - 0.5*(e.^2));
            nParams = size(b, 1);

            if IC == 1       % AIC
                ICC(k+1) = -2*npdf/T + 2*nParams/T;
            else             % BIC
                ICC(k+1) = -2*npdf/T + nParams*log(T)/T;
            end
        end

        [~, bestIdx] = min(ICC);
        beta = betaCell{bestIdx};
        eps = epsCell{bestIdx};
        lag = bestIdx - 1;

    else
        % --- Fixed lag ---
        k = maxlag;
        [X2, dy01] = buildDesignRes(const, dy, maxlag, T1, T, k);

        beta = X2 \ dy01;
        eps = dy01 - X2*beta;
        lag = maxlag;
    end
end

%% ---- Local function --------------------------------------------------------
function [X2, dy01] = buildDesignRes(const, dy, maxlag, T1, T, k)
% buildDesignRes  Construct restricted ADF design (no y_{t-1}).

    xx = const(maxlag+1:T1, :);
    dy01 = dy(maxlag+1:T1);

    if k > 0
        X2 = [xx, zeros(T, k)];
        for j = 1:k
            X2(:, size(xx, 2) + j) = dy(maxlag + 1 - j : T1 - j);
        end
    else
        X2 = xx;
    end
end
