function [ir, wold] = GetGirf(B, A0, nd, L, horz)
%GETGIRF Generalized impulse response functions

arguments
    B (:,:) double
    A0 (:,:) double
    nd (1,1) double {mustBeInteger, mustBeNonnegative}
    L (1,1) double {mustBeInteger, mustBePositive}
    horz (1,1) double {mustBeInteger, mustBeNonnegative}
end

rng(0, "twister");

N = size(B, 1);
B = quantecon.finance.dynamicnets.VarCompanion(B, nd, N, L);
J = [eye(N), zeros(N, N * (L - 1))];

ir = zeros(horz + 1, N, N);
wold = [];

for h = 0:horz
    wold = cat(3, wold, (J * (B^h) * J'));
end

for i = 1:N
    sv = false(1, N);
    sv(i) = true;
    ss = (1 / sqrt(A0(i, i)));
    for h = 1:horz + 1
        ir(h, :, i) = ss * sv * A0 * wold(:, :, h)';
    end
end

ir = permute(ir, [2 3 1]);
end
