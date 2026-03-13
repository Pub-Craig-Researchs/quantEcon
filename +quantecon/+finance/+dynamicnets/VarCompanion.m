function A = VarCompanion(A, ndet, n, p)
%VARCOMPANION Companion matrix for VAR coefficients

arguments
    A (:,:) double
    ndet (1,1) double {mustBeNonnegative}
    n (1,1) double {mustBeInteger, mustBePositive}
    p (1,1) double {mustBeInteger, mustBePositive}
end

rng(0, "twister");

A = A(:, ndet + 1:end);
A = [A; eye(n * (p - 1)), zeros(n * (p - 1), n)];
end
