classdef VarUtils
    %VARUTILS Shared utilities for VAR and Connectedness models

    methods (Static)
        function [X, Y] = var_setup(Data, p)
            % Prepare VAR design matrix
            [T, N] = size(Data);
            Y = Data(p+1:end, :);
            X = zeros(T-p, N*p+1);
            X(:, 1) = 1; % Constant
            for i = 1:p
                X(:, (i-1)*N+2 : i*N+1) = Data(p+1-i : end-i, :);
            end
        end

        function w = norm_kernel(T, H)
            % Generate Gaussian Kernel weights
            [J, I] = meshgrid(1:T, 1:T);
            Z = (I - J) ./ H;
            W_raw = (1/sqrt(2*pi)) * exp(-0.5 * Z.^2);
            sum_W = sum(W_raw, 2);
            w = W_raw ./ sum_W;
        end

        function ir = compute_girf(A_comp, Sigma, N, L, H)
            % Compute Generalized Impulse Response Functions
            % A_comp: N x N*p
            % Sigma: N x N
            F = zeros(N*L);
            F(1:N, :) = A_comp;
            if L > 1
                F(N+1:end, 1:N*(L-1)) = eye(N*(L-1));
            end

            J = [eye(N), zeros(N, N*(L-1))];

            Phi = zeros(N, N, H+1);
            F_pow = eye(N*L);

            for h = 0:H
                Phi(:, :, h+1) = J * F_pow * J';
                F_pow = F_pow * F;
            end

            ir = zeros(H+1, N, N);
            for j = 1:N
                shock = zeros(N, 1);
                shock(j) = 1;
                scale = 1 / sqrt(Sigma(j,j));
                pert = Sigma * shock * scale;
                for h = 0:H
                    ir(h+1, :, j) = Phi(:, :, h+1) * pert;
                end
            end
        end
    end
end
