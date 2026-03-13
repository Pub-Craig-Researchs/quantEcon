function results = DynamicNetworks(data, opts)
%DYNAMICNETWORKS Dynamic connectedness measures for time-varying networks
%
%   Usage:
%       res = quantecon.finance.DynamicNetworks(data)
%       res = quantecon.finance.DynamicNetworks(data, "NumSim", 100, "Lags", 1)
%
%   Inputs:
%       data - (T x N) matrix
%
%   Options:
%       NumSim      - Number of posterior simulations (default: 100)
%       Lags        - VAR lag length (default: 1)
%       KernelWidth - Kernel bandwidth (default: 8)
%       Horizon     - Wold horizon (default: 11)
%       Diagonal    - Use diagonal covariance (default: false)
%       Shrinkage   - Minnesota prior shrinkage (default: 0.05)
%       Seed        - RNG seed (default: 0)
%
%   References:
%       Barunik, J. and Ellington, M. (2020) Dynamic Networks in Large Financial
%       and Economic Systems.
%       Barunik, J. and Ellington, M. (2020) Dynamic Network Risk.

arguments
    data (:,:) double
    opts.NumSim (1,1) double {mustBeInteger, mustBePositive} = 100
    opts.Lags (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.KernelWidth (1,1) double {mustBePositive} = 8
    opts.Horizon (1,1) double {mustBeInteger, mustBePositive} = 11
    opts.Diagonal (1,1) logical = false
    opts.Shrinkage (1,1) double {mustBePositive} = 0.05
    opts.Seed (1,1) double = 0
end

rng(opts.Seed, "twister");

[T, N] = size(data);
L = opts.Lags;
nsim = opts.NumSim;

if T <= L
    error("quantecon:finance:DynamicNetworks:TooShort", "T must be larger than Lags.");
end

% Construct VAR design matrix
X = zeros(T - L, N * L);
for i = 1:L
    temp = quantecon.finance.dynamicnets.Lag0(data, i);
    X(:, 1 + N*(i-1) : i*N) = temp(1 + L:T, :);
end

y = data(1 + L:T, :);
T_eff = T - L;
X = [ones(T_eff, 1), X];

[SI, PI, a, RI] = quantecon.finance.dynamicnets.MinnesotaPrior(data, T, N, L, opts.Shrinkage);
priorprec0 = PI^(-1);

weights = quantecon.finance.dynamicnets.KernelWeights(T_eff, opts.KernelWidth);

HO = opts.Horizon;
diagonal = opts.Diagonal;

stab_ind = zeros(nsim, T_eff, "single");
max_eig = zeros(nsim, T_eff, "single");

TC_S = single(zeros(T_eff, nsim));
TC_M = single(zeros(T_eff, nsim));
TC_L = single(zeros(T_eff, nsim));
WC_S = TC_S; WC_M = TC_M; WC_L = TC_L;
TF_C = single(zeros(T_eff, nsim));
ND_L = single(zeros(T_eff, N, nsim));
ND_M = ND_L; ND_S = ND_L; TN_C = ND_L;
CT_L = ND_L; CT_M = ND_L; CT_S = ND_L; CT_T = ND_L;
CR_L = ND_L; CR_M = ND_L; CR_S = ND_L; CR_T = ND_L;

parfor kk = 1:T_eff
    w = weights(kk, :);
    bayesprec = (priorprec0 + X' * diag(w) * X);
    bayessv = bayesprec^(-1);
    BB = bayessv * ((X' * diag(w)) * y + priorprec0 * SI);
    bayesalpha = a + sum(w);

    g1 = SI' * priorprec0 * SI;
    g2 = y' * diag(w) * y;
    g3 = BB' * bayesprec * BB;
    bayesgamma = RI + g1 + g2 - g3;
    bayesgamma = 0.5 * bayesgamma + 0.5 * bayesgamma';

    wcs = zeros(1, nsim);
    wcm = wcs; wcl = wcs;
    tcs = wcs; tcm = wcs; tcl = wcs;
    tfc = zeros(1, nsim);
    ndcl = zeros(N, nsim); ndcm = ndcl; ndcs = ndcl; tndc = ndcl;
    ctcl = zeros(N, nsim); ctcm = ndcl; ctcs = ndcl; ctct = ndcl;
    crcl = zeros(N, nsim); crcm = ndcl; crcs = ndcl; crct = ndcl;

    for ii = 1:nsim
        mm = 0;
        while mm < 1
            SIGMA = iwishrnd(bayesgamma, bayesalpha);
            nu = randn(N*L + 1, N);
            Fi1 = (BB + chol(bayessv)' * nu * (chol(SIGMA)))';
            max_eig(ii, kk) = max(abs(eig([Fi1(:, 2:end); eye(N), zeros(N, N)])));
            if max_eig(ii, kk) < 0.999
                stab_ind(ii, kk) = 1;
                mm = 1;
            end
        end

        [~, wold] = quantecon.finance.dynamicnets.GetGirf(Fi1, SIGMA, 1, L, HO - 1);
        [ttfc, tc1, tc2, tc3, wc1, wc2, wc3, ct1, ct2, ct3, ctt, cr1, cr2, cr3, crt, ...
            ndc1, ndc2, ndc3, tnd1] = quantecon.finance.dynamicnets.GetDynnet(wold, T_eff, SIGMA, diagonal);

        wcs(:, ii) = wc3; wcm(:, ii) = wc2; wcl(:, ii) = wc1;
        tcs(:, ii) = tc3; tcm(:, ii) = tc2; tcl(:, ii) = tc1;
        tfc(:, ii) = ttfc;
        ndcl(:, ii) = ndc1; ndcm(:, ii) = ndc2; ndcs(:, ii) = ndc3; tndc(:, ii) = tnd1;
        ctcl(:, ii) = ct1; ctcm(:, ii) = ct2; ctcs(:, ii) = ct3; ctct(:, ii) = ctt;
        crcl(:, ii) = cr1; crcm(:, ii) = cr2; crcs(:, ii) = cr3; crct(:, ii) = crt;
    end

    WC_S(kk, :) = wcs; WC_M(kk, :) = wcm; WC_L(kk, :) = wcl;
    TC_S(kk, :) = tcs; TC_M(kk, :) = tcm; TC_L(kk, :) = tcl;
    TF_C(kk, :) = tfc;
    ND_S(kk, :, :) = ndcs; ND_M(kk, :, :) = ndcm; ND_L(kk, :, :) = ndcl;
    TN_C(kk, :, :) = tndc;
    CT_S(kk, :, :) = ctcs; CT_M(kk, :, :) = ctcm; CT_L(kk, :, :) = ctcl;
    CT_T(kk, :, :) = ctct;
    CR_S(kk, :, :) = crcs; CR_M(kk, :, :) = crcm; CR_L(kk, :, :) = crcl;
    CR_T(kk, :, :) = crct;
end

results = struct();
results.WithinShort = WC_S;
results.WithinMedium = WC_M;
results.WithinLong = WC_L;
results.TotalShort = TC_S;
results.TotalMedium = TC_M;
results.TotalLong = TC_L;
results.TotalFrequency = TF_C;
results.NetDirectionalShort = ND_S;
results.NetDirectionalMedium = ND_M;
results.NetDirectionalLong = ND_L;
results.NetDirectionalTotal = TN_C;
results.ToShort = CT_S;
results.ToMedium = CT_M;
results.ToLong = CT_L;
results.ToTotal = CT_T;
results.FromShort = CR_S;
results.FromMedium = CR_M;
results.FromLong = CR_L;
results.FromTotal = CR_T;
results.Stability.Indicator = stab_ind;
results.Stability.MaxEigenvalue = max_eig;
results.Options = opts;
end
