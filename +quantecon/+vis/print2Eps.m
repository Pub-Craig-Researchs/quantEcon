function print2Eps(name, fig, export_options, varargin)
%PRINT2EPS  Exports a figure to an EPS file
%
%   Ported to quantecon.vis.

options = {"-loose"};
if nargin > 3
    options = [options, varargin];
elseif nargin < 3
    export_options = 0;
    if nargin < 2
        fig = gcf();
    end
end

% Retrieve padding, crop & font-swap values
crop_amounts = nan(1,4);
if isstruct(export_options)
    try preserve_size = export_options.preserve_size; catch, preserve_size = false; end
    try fontswap      = export_options.fontswap;      catch, fontswap = true;       end
    try font_space    = export_options.font_space;    catch, font_space = "";       end
    font_space(2:end) = "";
    try bb_crop       = export_options.crop;          catch, bb_crop = 0;           end
    try crop_amounts  = export_options.crop_amounts;  catch,                        end
    try bb_padding    = export_options.bb_padding;    catch, bb_padding = 0;        end
    try renderer      = export_options.rendererStr;   catch, renderer = "-opengl";   end
    if ~isempty(renderer) && renderer(1) ~= "-", renderer = "-" + renderer; end
else
    preserve_size = numel(export_options) > 3 && export_options(4);
    fontswap = ~(numel(export_options) > 2) || export_options(3);
    bb_crop = 0; if numel(export_options) > 1, bb_crop = export_options(2); end
    bb_padding = 0; if numel(export_options) > 0, bb_padding = export_options(1); end
    renderer = "-opengl";
    font_space = "";
end

% Construct name
if isstring(name), name = char(name); end
if numel(name) < 5 || ~strcmpi(name(end-3:end), ".eps")
    name = [name, ".eps"];
end

% Ensure figure is visible for print
set(fig, "Visible", "on");
drawnow;

set(fig, "PaperPositionMode", "auto", "PaperOrientation", "portrait", "PaperUnits", "points");

font_handles = findall(fig, "-property", "FontName");
fonts = get(font_handles, "FontName");
if isempty(fonts), fonts = {}; elseif ~iscell(fonts), fonts = {fonts}; end

fontsl = lower(fonts);
for a = 1:numel(fonts)
    f = fontsl{a};
    f(f == " ") = [];
    switch f
        case {"times", "timesnewroman", "times-roman"}
            fontsl{a} = "times";
    end
end

black_text_handles = findall(fig, "Type", "text", "Color", [0 0 0]);
white_text_handles = findall(fig, "Type", "text", "Color", [1 1 1]);
set(black_text_handles, "Color", [0 0 0] + eps);
set(white_text_handles, "Color", [1 1 1] - eps);
white_line_handles = findall(fig, "Type", "line", "Color", [1 1 1]);
set(white_line_handles, "Color", [1 1 1] - 0.00001);

hAxes = findall(fig, "Type", "axes");
if quantecon.vis.usingHg2(fig) && ~isempty(hAxes)
    try
        oldSortMethods = get(hAxes, {"SortMethod"});
        if any(~strcmpi("ChildOrder", oldSortMethods))
            imgBefore = quantecon.vis.print2Array(fig);
            set(hAxes, "SortMethod", "ChildOrder");
            imgAfter  = quantecon.vis.print2Array(fig);
            if ~isequal(imgBefore, imgAfter)
                set(hAxes, {"SortMethod"}, oldSortMethods);
            end
        end
    catch
    end
end

options{end+1} = "-depsc2";
if ~quantecon.vis.usingHg2(fig), fig = double(fig); end

if quantecon.vis.usingHg2(fig)
    origAlphaColors = eps_maintainAlpha(fig);
end

drawnow; pause(0.05);
print(fig, options{:}, char(name));

try set(hAxes, {"SortMethod"}, oldSortMethods); catch, end

try
    fstrm = quantecon.vis.readWriteEntireTextfile(name);
catch
    fstrm = "";
end

if quantecon.vis.usingHg2(fig) && ~isempty(fstrm)
    [~, fstrm, foundFlags] = eps_maintainAlpha(fig, fstrm, origAlphaColors);
    if ~all(foundFlags)
    end
end

if isempty(fstrm), return; end

if quantecon.vis.usingHg2(fig)
    fstrm = strrep(fstrm, sprintf("\n10.0 ML\n"), sprintf("\n1 LJ\n"));
    fstrm = strrep(fstrm, sprintf("GC\n2 setlinecap\n1 LJ\nN"), sprintf("GC\n2 setlinecap\n1 LJ\n0.667 LW\nN"));
    fstrm = strrep(fstrm, sprintf("\n2 setlinecap\n1 LJ\nN"), sprintf("\n2 setlinecap\n1 LJ\n1 LW\nN"));
end

try set(black_text_handles, "Color", [0 0 0]); set(white_text_handles, "Color", [1 1 1]); catch, end
set(white_line_handles, "Color", [1 1 1]);

if preserve_size
    paper_size = get(fig, "PaperSize");
    fstrm = sprintf("<< /PageSize [%d %d] >> setpagedevice\n%s", paper_size, fstrm);
end

quantecon.vis.readWriteEntireTextfile(name, fstrm);
end

function [StoredColors, fstrm, foundFlags] = eps_maintainAlpha(fig, fstrm, StoredColors)
if nargin == 1
    hObjs = findall(fig);
    StoredColors = {};
    propNames = {"Face", "Edge"};
    for objIdx = 1:length(hObjs)
        hObj = hObjs(objIdx);
        for propIdx = 1:numel(propNames)
            try
                propName = propNames{propIdx};
                if strcmp(hObj.(propName).ColorType, "truecoloralpha")
                    oldColor = hObj.(propName).ColorData;
                    if numel(oldColor) > 3 && oldColor(4) ~= 255
                        nColors = length(StoredColors);
                        newColor = uint8([101; 102+floor(nColors/255); mod(nColors, 255); 255]);
                        StoredColors{end+1} = {hObj, propName, oldColor, newColor};
                        hObj.(propName).ColorData = newColor;
                    end
                end
            catch
            end
        end
    end
else
    nColors = length(StoredColors);
    foundFlags = false(1, nColors);
    for objIdx = 1:nColors
        colorsData = StoredColors{objIdx};
        hObj = colorsData{1}; propName = colorsData{2}; origColor = colorsData{3}; newColor = colorsData{4};
        try
            colorID = num2str(round(double(newColor(1:3)') / 255, 3), "%.3g %.3g %.3g");
            origRGB = num2str(round(double(origColor(1:3)') / 255, 3), "%.3g %.3g %.3g");
            origAlpha = num2str(round(double(origColor(end)) / 255, 3), "%.3g");
            newStr = sprintf("\n%s RC\n%s .setopacityalpha true\n", origRGB, origAlpha);
            oldStr = sprintf("\n%s RC\n", colorID);
            foundFlags(objIdx) = ~isempty(strfind(fstrm, oldStr));
            fstrm = strrep(fstrm, oldStr, newStr);
            hObj.(propName).ColorData = origColor;
        catch
        end
    end
end
end
