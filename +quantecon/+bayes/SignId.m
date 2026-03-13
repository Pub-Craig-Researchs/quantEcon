classdef SignId
    %SIGNID Engine for Bayesian Sign Restrictions
    %   Algorithm: Pure rotation (Rubio-Ramirez et al., 2010)

    methods (Static)
        function [struct_irfs, Q_record] = identify(coeffs, sigma, p, horizon, restrictions, nsamp)
            % IDENTIFY structural shocks via sign restrictions
            % restrictions: cell array {N x N} containing 1, -1, or 0

            N = size(coeffs, 2);
            % Initialize struct_irfs with zeros for each variable-shock pair
            struct_irfs = cell(N, N);
            for var_idx = 1:N
                for shock_idx = 1:N
                    struct_irfs{var_idx, shock_idx} = zeros(nsamp, horizon);
                end
            end
            Q_record = zeros(N, N, nsamp);

            % 1. Get Reduced Form IRFs
            reduced_irfs = quantecon.bayes.BvarAnalysis.irf(coeffs, sigma, p, horizon, "none");
            C0 = chol(sigma, 'lower');

            success = 0;
            attempts = 0;
            while success < nsamp && attempts < nsamp * 100
                attempts = attempts + 1;
                % 2. Draw Random Rotation Q
                [Q, ~] = qr(randn(N));
                % Ensure unique and proper rotation
                Q = Q * diag(sign(diag(Q)));

                % 3. Check Restrictions on Impact (Horizon 1)
                D = C0 * Q;
                if check_restrictions(D, restrictions)
                    success = success + 1;
                    Q_record(:, :, success) = Q;
                    % Map all horizons
                    for h = 1:horizon
                        Rh = reduced_irfs(:, :, h) * D;
                        for var = 1:N
                            for shock = 1:N
                                struct_irfs{var, shock}(success, h) = Rh(var, shock);
                            end
                        end
                    end
                end
            end

            if success < nsamp
                warning('Only %d successful rotations found after %d attempts.', success, attempts);
            end
        end
    end
end

function ok = check_restrictions(D, R)
% D is N x N impact matrix
% R is N x N restriction matrix (1: pos, -1: neg, 0: none)
[N, ~] = size(D);
ok = true;
for i = 1:N
    for j = 1:N
        if R(i,j) == 1 && D(i,j) < 0; ok = false; return; end
        if R(i,j) == -1 && D(i,j) > 0; ok = false; return; end
    end
end
end
