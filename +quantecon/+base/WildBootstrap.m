function boot_samples = WildBootstrap(data, nboot, opts)
%WILDBOOTSTRAP Perform Wild Bootstrap resampling
%
%   Usage:
%       samples = quantecon.base.WildBootstrap(residuals, 1000)
%       samples = quantecon.base.WildBootstrap(residuals, 1000, 'WeightType', 'Rademacher')
%
%   Inputs:
%       data   - (T x K) Matrix to resample (usually residuals)
%       nboot  - (int) Number of bootstrap replicates
%
%   Options:
%       'WeightType' - 'Rademacher' (Default) or 'Normal'
%
%   Outputs:
%       boot_samples - (T x K x nboot) 3D array of resampled data

arguments
    data (:,:) double {mustBeNumeric, mustBeReal}
    nboot (1,1) double {mustBeInteger, mustBePositive} = 1000
    opts.WeightType (1,1) string {mustBeMember(opts.WeightType, ["Rademacher", "Normal"])} = "Rademacher"
end

[T, K] = size(data);
boot_samples = zeros(T, K, nboot);

rng(0, "twister"); % For replicability within the call

for b = 1:nboot
    if opts.WeightType == "Rademacher"
        % Weights are 1 or -1 with probability 0.5
        w = (rand(T, 1) < 0.5) * 2 - 1;
    else
        % Weights are N(0,1)
        w = randn(T, 1);
    end

    boot_samples(:, :, b) = data .* w;
end
end
