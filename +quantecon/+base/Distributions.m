classdef Distributions
    %DISTRIBUTIONS Core distribution helpers for Bayesian econometrics.

    methods (Static)
        function A = wish(h, n)
            %WISH Draw from a Wishart distribution.
            %   A = Distributions.wish(H, N) draws an m x m matrix from a
            %   Wishart distribution with scale matrix H and degrees of
            %   freedom NU = N.
            %
            %   Note: Parameterized so that mean is N*H.

            arguments
                h (:,:) double
                n (1,1) double
            end

            % Bartlett's decomposition
            A = chol(h)' * randn(size(h, 1), n);
            A = A * A';
        end

        function [logval, val] = mgamma(a, n)
            %MGAMMA Multivariate gamma function.
            %   [logval, val] = Distributions.mgamma(A, N) computes the
            %   log-value and value of the multivariate gamma function.

            arguments
                a (1,1) double
                n (1,1) double
            end

            const = (n * (n - 1) / 4) * log(pi);
            temp = zeros(n, 1);
            for jj = 1:n
                temp(jj) = gammaln(a + 0.5 * (1 - jj));
            end

            logval = const + sum(temp);
            val = exp(logval);
        end

        function draw = igamma(alpha, beta)
            %IGAMMA Draw from an Inverse Gamma distribution.
            %   draw = Distributions.igamma(ALPHA, BETA)

            arguments
                alpha (1,1) double
                beta (1,1) double
            end

            draw = 1 ./ gamrnd(alpha, 1./beta);
        end
    end
end
