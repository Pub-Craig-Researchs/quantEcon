function [TFC, TC1, TC2, TC3, WC1, WC2, WC3, DCTL, DCTM, DCTS, DCTT, DCRL, DCRM, DCRS, DCRT, NDCL, NDCM, NDCS, TNDC] = GetDynnet(wo, TT, sig, diagonal)
%GETDYNNET Frequency connectedness decomposition

arguments
    wo (:,:,:) double
    TT (1,1) double {mustBeInteger, mustBePositive}
    sig (:,:) double
    diagonal (1,1) logical
end

rng(0, "twister");

Tw = floor(TT / 10);
omeg = linspace(0, pi, Tw)';

omeg2 = pi ./ omeg;
bandmat = [omeg, omeg2];

longIdx = bandmat(:, 2) > 20;
medIdx = bandmat(:, 2) <= 20 & bandmat(:, 2) > 5;
shortIdx = bandmat(:, 2) <= 5;

d1 = sum(longIdx);
d2 = sum(medIdx);
d3 = sum(shortIdx);

HO = size(wo, 3);
N = size(wo, 1);

DCRL = zeros(1, N);
DCTL = zeros(1, N);
DCRM = zeros(1, N);
DCTM = zeros(1, N);
DCRS = zeros(1, N);
DCTS = zeros(1, N);
DCRT = zeros(1, N);
DCTT = zeros(1, N);

Omeg = zeros(N, N, length(omeg));

GI = zeros(N, N, length(omeg));

for w = 1:length(omeg)
    for nn = 1:HO
        GI(:, :, w) = GI(:, :, w) + wo(:, :, nn) * exp(-1i * nn * omeg(w));
    end
    if ~diagonal
        Omeg(:, :, w) = GI(:, :, w) * sig * GI(:, :, w)';
    else
        Omeg(:, :, w) = GI(:, :, w) * diag(diag(sig)) * GI(:, :, w)';
    end
end
Omeg = sum(real(Omeg), 3);

FC = zeros(N, N, length(omeg));
for w = 1:length(omeg)
    if ~diagonal
        GI1 = GI(:, :, w) * sig;
    else
        GI1 = GI(:, :, w) * diag(diag(sig));
    end
    PS = zeros(N, N);
    PP = zeros(N, N);
    for k = 1:N
        for j = 1:N
            PS(j, k) = (abs(GI1(j, k)))^2;
            PP(j, k) = PS(j, k) / (Omeg(j, j) * sig(k, k));
        end
    end
    FC(:, :, w) = PP;
end

PP1 = sum(FC, 3);
for w = 1:length(omeg)
    for j = 1:N
        FC(j, :, w) = FC(j, :, w) ./ sum(PP1(j, :));
    end
end

thetainf = sum(FC, 3);

% Long-term
if d1 > 0
    temp1 = sum(FC(:, :, 1:d1), 3);
else
    temp1 = zeros(N);
end

tr = zeros(1, N);
tt = zeros(1, N);
for i = 1:N
    tr(i) = sum(temp1(i, :)) - temp1(i, i);
    tt(i) = sum(temp1(:, i)) - temp1(i, i);
    DCRL(i, :) = sum(temp1(i, :));
    DCTL(i, :) = sum(temp1(:, i));
end
NDCL = (tt - tr);
WC1 = 100 * (1 - trace(temp1) / sum(temp1, "all"));
TC1 = WC1 * (sum(temp1, "all") / sum(thetainf, "all"));

% Medium-term
if d3 > 0
    temp1 = sum(FC(:, :, d1 + 1:length(omeg) - d3), 3);
else
    temp1 = sum(FC(:, :, d1 + 1:end), 3);
end

tr = zeros(1, N);
tt = zeros(1, N);
for i = 1:N

    tr(i) = sum(temp1(i, :)) - temp1(i, i);
    tt(i) = sum(temp1(:, i)) - temp1(i, i);
    DCRM(i, :) = sum(temp1(i, :));
    DCTM(i, :) = sum(temp1(:, i));
end
NDCM = (tt - tr);
WC2 = 100 * (1 - trace(temp1) / sum(temp1, "all"));
TC2 = WC2 * (sum(temp1, "all") / sum(thetainf, "all"));

% Short-term
temp1 = sum(FC(:, :, d1 + d2 + 1:end), 3);
tr = zeros(1, N);
tt = zeros(1, N);
for i = 1:N

    tr(i) = sum(temp1(i, :)) - temp1(i, i);
    tt(i) = sum(temp1(:, i)) - temp1(i, i);
    DCRS(i, :) = sum(temp1(i, :));
    DCTS(i, :) = sum(temp1(:, i));
end
NDCS = (tt - tr);
WC3 = 100 * (1 - trace(temp1) / sum(temp1, "all"));
TC3 = WC3 * (sum(temp1, "all") / sum(thetainf, "all"));

TFC = TC1 + TC2 + TC3;

temp1 = sum(FC, 3);
tr = zeros(1, N);
tt = zeros(1, N);
for i = 1:N

    tr(i) = sum(temp1(i, :)) - temp1(i, i);
    tt(i) = sum(temp1(:, i)) - temp1(i, i);
    DCRT(i, :) = sum(temp1(i, :));
    DCTT(i, :) = sum(temp1(:, i));
end
TNDC = (tt - tr);

tfc1 = 100 * (sum(temp1, "all") / sum(thetainf, "all") - trace(temp1) / sum(thetainf, "all"));
if abs(TFC - tfc1) > eps + 1.0e-10
    error("quantecon:finance:dynamicnets:ConsistencyError", "Frequency connectedness check failed.");
end

DCRL = DCRL(:)';
DCRM = DCRM(:)';
DCRS = DCRS(:)';
DCTL = DCTL(:)';
DCTM = DCTM(:)';
DCTS = DCTS(:)';
DCTT = DCTT(:)';
DCRT = DCRT(:)';
end
