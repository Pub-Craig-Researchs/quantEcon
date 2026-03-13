function out = Lag0(x, p)
%LAG0 Lag variables by p with zero padding

arguments
    x (:,:) double
    p (1,1) double {mustBeInteger, mustBeNonnegative}
end

rng(0, "twister");

[R, C] = size(x);
if p == 0
    out = x;
    return;
end

x1 = x(1:(R - p), :);
out = [zeros(p, C); x1];
end
