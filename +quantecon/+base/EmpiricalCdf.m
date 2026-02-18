function u = EmpiricalCdf(data)
%EMPIRICALCDF Probability Integral Transform using Empirical CDF
%
%   Usage:
%       u = quantecon.base.EmpiricalCdf(data)
%
%   Input:
%       data - (T x K) matrix of data
%
%   Output:
%       u - (T x K) matrix of uniform margins in (0,1)

arguments
    data (:,:) double {mustBeNumeric, mustBeReal}
end

[T, K] = size(data);
u = zeros(T, K);

for k = 1:K
    % Efficient ranking for ECDF
    % Use T+1 as denominator to avoid exactly 1.0 (problematic for some copulas)
    [~, idx] = sort(data(:, k));
    [~, rank] = sort(idx);
    u(:, k) = rank / (T + 1);
end
end
