function adjw = KernelWeights(T, H)
%KERNELWEIGHTS Normal kernel weights with normalization

arguments
    T (1,1) double {mustBeInteger, mustBePositive}
    H (1,1) double {mustBePositive}
end

rng(0, "twister");

ww = zeros(T, T);
for i = 1:T
    for j = 1:T
        z = (i - j) / H;
        ww(i, j) = (1 / sqrt(2*pi)) * exp(-0.5 * (z^2));
    end
end

s = sum(ww, 2);
adjw = zeros(T, T);
for k = 1:T
    adjw(k, :) = ww(k, :) / s(k);
end

cons = sum(adjw.^2, 2);
for k = 1:T
    adjw(k, :) = (1 / cons(k)) * adjw(k, :);
end
end
