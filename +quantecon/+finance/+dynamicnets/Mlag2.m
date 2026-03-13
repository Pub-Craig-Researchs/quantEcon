function Xlag = Mlag2(X, p)
%MLAG2 Create lagged design matrix

arguments
    X (:,:) double
    p (1,1) double {mustBeInteger, mustBePositive}
end

rng(0, "twister");

[Traw, N] = size(X);
Xlag = zeros(Traw, N * p);
for ii = 1:p
    Xlag(p + 1:Traw, (N * (ii - 1) + 1):N * ii) = X(p + 1 - ii:Traw - ii, 1:N);
end
end
