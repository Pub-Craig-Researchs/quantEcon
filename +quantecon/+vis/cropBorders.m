function [A, vA, vB, bb_rel] = cropBorders(A, bcol, padding, crop_amounts)
%CROPBORDERS Crop the borders of an image or stack of images
%
%   Ported to quantecon.vis namespace.

if nargin < 3, padding = 0; end
if nargin < 4, crop_amounts = nan(1,4); end
crop_amounts(end+1:4) = NaN;

[h, w, c, n] = size(A);
if isempty(bcol)
    bcol = A(ceil(end/2), 1, :, 1);
end
if isscalar(bcol)
    bcol = bcol(ones(c, 1));
end

% Crop margin from left
if ~isfinite(crop_amounts(4))
    bail = false;
    for l = 1:w
        for a = 1:c
            if ~all(col(A(:,l,a,:)) == bcol(a))
                bail = true;
                break;
            end
        end
        if bail, break; end
    end
else
    l = 1 + abs(crop_amounts(4));
end

% Crop margin from right
if ~isfinite(crop_amounts(2))
    bcol_r = A(ceil(end/2), w, :, 1);
    bail = false;
    for r = w:-1:l
        for a = 1:c
            if ~all(col(A(:,r,a,:)) == bcol_r(a))
                bail = true;
                break;
            end
        end
        if bail, break; end
    end
else
    r = w - abs(crop_amounts(2));
end

% Crop margin from top
if ~isfinite(crop_amounts(1))
    bcol_t = A(1, ceil(end/2), :, 1);
    bail = false;
    for t = 1:h
        for a = 1:c
            if ~all(col(A(t,:,a,:)) == bcol_t(a))
                bail = true;
                break;
            end
        end
        if bail, break; end
    end
else
    t = 1 + abs(crop_amounts(1));
end

% Crop margin from bottom
bcol_b = A(h, ceil(end/2), :, 1);
if ~isfinite(crop_amounts(3))
    bail = false;
    for b = h:-1:t
        for a = 1:c
            if ~all(col(A(b,:,a,:)) == bcol_b(a))
                bail = true;
                break;
            end
        end
        if bail, break; end
    end
else
    b = h - abs(crop_amounts(3));
end

if padding == 0
    % No padding
elseif abs(padding) < 1
    padding = sign(padding) * round(mean([b-t, r-l]) * abs(padding));
else
    padding = round(padding);
end

if padding > 0
    B = repmat(bcol_b, [(b-t)+1+padding*2, (r-l)+1+padding*2, 1, n]);
    vA = [t, b, l, r];
    vB = [padding+1, (b-t)+1+padding, padding+1, (r-l)+1+padding];
    B(vB(1):vB(2), vB(3):vB(4), :, :) = A(vA(1):vA(2), vA(3):vA(4), :, :);
    A = B;
else
    vA = [MAX(1, t-padding), MIN(h, b+padding), MAX(1, l-padding), MIN(w, r+padding)];
    A = A(vA(1):vA(2), vA(3):vA(4), :, :);
    vB = [NaN, NaN, NaN, NaN];
end

bb_pixels = [l-1, h-b-1, r+1, h-t+1];
bb_pixels(bb_pixels < 0) = 0;
bb_pixels = min(bb_pixels, [w, h, w, h]);
bb_rel = bb_pixels ./ [w, h, w, h];
end

function A = col(A)
A = A(:);
end

function val = MAX(a, b), val = max(a, b); end
function val = MIN(a, b), val = min(a, b); end
