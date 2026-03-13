function bands = HpdInterval(draws, opts)
%HPDINTERVAL Highest Posterior Density (HPD) interval
%
%   Computes the shortest credible interval containing a specified
%   probability mass from posterior draws.
%
%   Reference:
%       Chen, M.-H. & Shao, Q.-M. (1999). "Monte Carlo Estimation of
%       Bayesian Credible and HPD Intervals." J. Comp. Graph. Stat. 8.
%
%   Usage:
%       bands = quantecon.bayes.HpdInterval(draws);
%       bands = quantecon.bayes.HpdInterval(draws, 'Prob', 0.9, 'Shortest', true);
%
%   Inputs:
%       draws - (nDraws x D) matrix of posterior draws
%
%   Options:
%       'Prob'     - (double) coverage probability (Default: 0.90)
%       'Shortest' - (bool)   if true, find the shortest interval;
%                    if false, chop symmetric tails (Default: true)
%
%   Outputs:
%       bands - (2 x D) matrix; row 1 = lower bound, row 2 = upper bound

arguments
    draws (:,:) double
    opts.Prob     (1,1) double {mustBeInRange(opts.Prob, 0, 1, 'exclusive')} = 0.90
    opts.Shortest (1,1) logical = true
end

[nDraws, D] = size(draws);
bands  = zeros(2, D);
nWidth = round(opts.Prob * nDraws);

for d = 1:D
    col = sort(draws(:, d), 'descend');     % descending

    if opts.Shortest
        % Slide a window of width nWidth, pick narrowest
        minW = col(1) - col(nWidth);
        bup  = 1;
        for j = 2:(nDraws - nWidth + 1)
            w = col(j) - col(j + nWidth - 1);
            if w < minW
                minW = w;
                bup  = j;
            end
        end
    else
        % Symmetric tail chop
        bup = nDraws - nWidth - floor(0.5 * (nDraws - nWidth));
    end

    bands(2, d) = col(bup);               % upper
    bands(1, d) = col(bup + nWidth - 1);   % lower
end
end
