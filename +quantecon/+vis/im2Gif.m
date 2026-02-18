function im2Gif(A, varargin)
%IM2GIF Convert a multiframe image to an animated GIF file
%
%   Ported to quantecon.vis.

[A, options] = parseArgs(A, varargin{:});

if options.crop
    A = quantecon.vis.cropBorders(A, A(ceil(end/2), 1, :, 1));
end

[h, w, c, n] = size(A);
uns = cell(1, size(A, 4));
for nn = 1:size(A, 4)
    uns{nn} = unique(reshape(A(:,:,:,nn), h*w, c), "rows");
end
map = unique(cell2mat(uns'), "rows");

A_flat = reshape(permute(A, [1, 2, 4, 3]), h, w*n, c);

if size(map, 1) > 256
    dither_str = "dither";
    if ~options.dither, dither_str = "nodither"; end

    if options.ncolors <= 1
        [B, map] = rgb2ind(A_flat, options.ncolors, dither_str);
        if size(map, 1) > 256, [B, map] = rgb2ind(A_flat, 256, dither_str); end
    else
        [B, map] = rgb2ind(A_flat, min(round(options.ncolors), 256), dither_str);
    end
else
    if max(map(:)) > 1
        map = double(map) / 255;
        A_flat = double(A_flat) / 255;
    end
    B = rgb2ind(im2double(A_flat), map);
end
B = reshape(B, h, w, 1, n);
map(B(1)+1, :) = im2double(A_flat(1, 1, :));

imwrite(B, map, options.outfile, "LoopCount", round(options.loops(1)), "DelayTime", options.delay);
end

function [A, options] = parseArgs(A, varargin)
options = struct("outfile", "", "dither", true, "crop", true, "ncolors", 256, "loops", 65535, "delay", 1/15);
a = 0; n = numel(varargin);
while a < n
    a = a + 1;
    if ischar(varargin{a}) && ~isempty(varargin{a})
        if varargin{a}(1) == "-"
            opt = lower(varargin{a}(2:end));
            switch opt
                case "nocrop"
                    options.crop = false;
                case "nodither"
                    options.dither = false;
                otherwise
                    if isfield(options, opt)
                        a = a + 1;
                        if ischar(varargin{a}) && ~ischar(options.(opt)), options.(opt) = str2double(varargin{a});
                        else, options.(opt) = varargin{a}; end
                    end
            end
        else
            options.outfile = varargin{a};
        end
    end
end

if isempty(options.outfile)
    if ~ischar(A), error("No output filename given."); end
    [path, outfile] = fileparts(A);
    options.outfile = fullfile(path, [outfile, ".gif"]);
end

if ischar(A), A = imreadRgb(A); end
end

function A = imreadRgb(name)
info = imfinfo(name);
switch lower(info(1).Format)
    case "gif"
        [A, map] = imread(name, "frames", "all");
        if ~isempty(map)
            map = uint8(map * 256 - 0.5);
            A = reshape(map(uint32(A)+1, :), [size(A), size(map, 2)]);
            A = permute(A, [1, 2, 5, 4, 3]);
        end
    case {"tif", "tiff"}
        A_cell = cell(numel(info), 1);
        for a = 1:numel(A_cell)
            [img, map] = imread(name, "Index", a, "Info", info);
            if ~isempty(map)
                map = uint8(map * 256 - 0.5);
                img = reshape(map(uint32(img)+1,:), [size(img), size(map, 2)]);
            end
            if size(img, 3) == 4
                img = single(img); img = 255 - img;
                img = uint8(img(:,:,1:3) .* (img(:,:,4)/255));
            elseif size(img, 3) < 3
                img = cat(3, img, img, img);
            end
            A_cell{a} = img;
        end
        A = cat(4, A_cell{:});
    otherwise
        [A, map, ~] = imread(name);
        A = A(:,:,:,1);
        if ~isempty(map)
            map = uint8(map * 256 - 0.5);
            A = reshape(map(uint32(A)+1, :), [size(A), size(map, 2)]);
        end
end
end
