function lnprior = PriorDens(para, pmean, pstdd, pshape)
%PRIORDENS Evaluate log prior density for DSGE model parameters.
%
%   Computes the sum of log prior densities for a vector of structural
%   parameters. Supports Beta, Gamma, Normal, Inverse-Gamma, and Uniform.
%
%   INPUTS:
%       para   - (K x 1) parameter values
%       pmean  - (K x 1) prior means (or shape parameter a for InvGamma,
%                left bound for Uniform)
%       pstdd  - (K x 1) prior std devs (or shape parameter b for InvGamma,
%                right bound for Uniform)
%       pshape - (K x 1) distribution codes:
%                  0 = point mass (parameter fixed, ignored in density)
%                  1 = Beta(mean, stdd)
%                  2 = Gamma(mean, stdd)
%                  3 = Normal(mean, stdd)
%                  4 = Inverse-Gamma(s^2, nu)
%                  5 = Uniform(a, b)
%
%   OUTPUT:
%       lnprior - (scalar) log prior density value
%
%   Reference:
%       Standard Bayesian DSGE estimation priors. Originally from
%       Del Negro & Schorfheide (2004) / NY Fed DSGE codebase.
%
%   See also: quantecon.dsge.Gensys, quantecon.bayes.MhSampler

arguments
    para   (:,1) double
    pmean  (:,1) double
    pstdd  (:,1) double
    pshape (:,1) double
end

K = length(pshape);
assert(length(para) == K && length(pmean) == K && length(pstdd) == K, ...
    'All inputs must have the same length.');

lnprior = 0;

for i = 1:K
    switch pshape(i)
        case 0
            % Point mass: contributes nothing to log density
            continue

        case 1  % Beta prior
            % Parameterise Beta(a,b) from mean and std
            mu = pmean(i);
            sd = pstdd(i);
            a = (1 - mu) * mu^2 / sd^2 - mu;
            b = a * (1/mu - 1);
            lnprior = lnprior + (a-1)*log(para(i)) + (b-1)*log(1-para(i)) - betaln(a, b);

        case 2  % Gamma prior
            mu = pmean(i);
            sd = pstdd(i);
            b = sd^2 / mu;        % scale
            a = mu / b;            % shape
            lnprior = lnprior + (a-1)*log(para(i)) - para(i)/b - gammaln(a) - a*log(b);

        case 3  % Normal prior
            mu = pmean(i);
            sd = pstdd(i);
            lnprior = lnprior - 0.5*log(2*pi) - log(sd) - 0.5*((para(i)-mu)/sd)^2;

        case 4  % Inverse-Gamma prior
            % para ~ IG(s^2, nu): p(x) proportional to x^{-(nu+1)} exp(-nu*s^2/(2*x^2))
            s = pmean(i);   % scale
            nu = pstdd(i);  % shape (degrees of freedom)
            lnprior = lnprior + log(2) - gammaln(nu/2) + (nu/2)*log(nu*s^2/2) ...
                      - ((nu+1)/2)*log(para(i)^2) - nu*s^2/(2*para(i)^2);

        case 5  % Uniform prior
            a = pmean(i);  % left bound
            b = pstdd(i);  % right bound
            if para(i) >= a && para(i) <= b
                lnprior = lnprior - log(b - a);
            else
                lnprior = -Inf;
                return
            end

        otherwise
            error('PriorDens:unknownShape', 'Unknown prior shape code %d for parameter %d.', pshape(i), i);
    end
end

end
