function [SI, PI, a, RI] = MinnesotaPrior(Y, T, N, L, shrinkage)
%MINNESOTAPRIOR Minnesota prior specification for TVP-VAR

arguments
    Y (:,:) double
    T (1,1) double {mustBeInteger, mustBePositive}
    N (1,1) double {mustBeInteger, mustBePositive}
    L (1,1) double {mustBeInteger, mustBePositive}
    shrinkage (1,1) double {mustBePositive}
end

rng(0, "twister");

K = N * L + 1;
SI = [zeros(1, N); 0.1 * eye(N); zeros((L - 1) * N, N)];
PI = zeros(K, 1);

sigma_sq = zeros(N, 1);
for i = 1:N
    Y_i = quantecon.finance.dynamicnets.Mlag2(Y(:, i), L);
    Y_i = Y_i(L + 1:T, :);
    X_i = [ones(T - L, 1), Y_i];
    y_i = Y(L + 1:T, i);
    alpha_i = (X_i' * X_i) \ (X_i' * y_i);
    sigma_sq(i, 1) = (1 / (T - L + 1)) * (y_i - X_i * alpha_i)' * (y_i - X_i * alpha_i);
end

s = sigma_sq.^(-1);
for ii = 1:L
    PI(2 + N*(ii - 1) : 1 + N*ii) = (shrinkage^2) * s / (ii^2);
end
PI(1) = 10^2;
PI = diag(PI);

% Wishart priors
a = max(N + 2, N + 2 * 8 - T);
RI = (a - N - 1) * sigma_sq;
RI = diag(RI);
end
