function [A, bcol, alpha] = print2Array(fig, res, renderer, gs_options)
%PRINT2ARRAY  Exports a figure to a bitmap RGB image array
%
%   Ported to quantecon.vis.

if nargin < 1, fig = gcf; end
if nargin < 2, res = 1; end

drawnow;
old_mode = get(fig, "Units");
set(fig, "Units", "pixels");
px = get(fig, "Position");
set(fig, "Units", old_mode);

pause(0.05);

bcol = get(fig, "Color");
try
    if res == 1
        [A, alpha] = getJavaImage(fig);
    else
        error("magnify/downscale via print() to image file and then import");
    end
catch
    npx = prod(px(3:4) * res) / 1e6;
    if npx > 30
        warning("MATLAB:LargeImage", "print2Array generating a %.1fM pixel image. This could be slow.", npx);
    end
    res_str = "-r" + num2str(ceil(get(0, "ScreenPixelsPerInch") * res));
    tmp_nam = [tempname, ".tif"];
    try
        fid = fopen(tmp_nam, "w");
        fwrite(fid, 1);
        fclose(fid);
        delete(tmp_nam);
        isTempDirOk = true;
    catch
        [~, fname, fext] = fileparts(tmp_nam);
        fpath = pwd;
        tmp_nam = fullfile(fpath, [fname, fext]);
        isTempDirOk = false;
    end

    isRetry = false;
    if nargin > 3 && ~isempty(gs_options)
        if isequal(gs_options, "retry")
            isRetry = true;
            gs_options = "";
        elseif iscell(gs_options)
            gs_options = sprintf(" %s", gs_options{:});
        elseif ~ischar(gs_options)
            error("gs_options input argument must be a string or cell-array of strings");
        else
            gs_options = " " + gs_options;
        end
    else
        gs_options = "";
    end

    if nargin > 2 && strcmp(renderer, "-painters")
        try
            [A, alpha, err, ex] = getPrintImage(fig, res_str, renderer, tmp_nam);
            if err, rethrow(ex); end
        catch
            if isTempDirOk
                tmp_eps = [tempname, ".eps"];
            else
                tmp_eps = fullfile(fpath, [fname, ".eps"]);
            end
            quantecon.vis.print2Eps(tmp_eps, fig, 0, renderer, "-loose");
            try
                cmd_str = "-dEPSCrop -q -dNOPAUSE -dBATCH " + res_str + " -sDEVICE=tiff24nc";
                fp = fontPath();
                if ~isempty(fp)
                    cmd_str = cmd_str + ' -sFONTPATH="' + fp + '"';
                end
                cmd_str = cmd_str + ' -sOutputFile="' + tmp_nam + '" "' + tmp_eps + '"' + gs_options;
                quantecon.vis.ghostscript(cmd_str);
            catch me
                delete(tmp_eps);
                rethrow(me);
            end
            delete(tmp_eps);
            A = imread(tmp_nam);
            delete(tmp_nam);
        end
    else
        if nargin < 3, renderer = "-opengl"; end
        if ~isempty(renderer) && renderer(1) ~= "-", renderer = "-" + renderer; end
        [A, alpha, err, ex] = getPrintImage(fig, res_str, renderer, tmp_nam);
        if err
            if ~isRetry
                fprintf(2, "An error occurred in Matlab's print function:\n%s\n", ex.message);
            end
            rethrow(ex);
        end
    end
end

if isequal(bcol, "none")
    bcol = squeeze(A(1, 1, :));
    if ~all(bcol == 0), bcol = [255, 255, 255]; end
else
    if all(bcol <= 1), bcol = bcol * 255; end
    if ~isequal(bcol, round(bcol))
        bcol = squeeze(A(1, 1, :));
    end
end
bcol = uint8(bcol);

if isequal(res, round(res))
    px_sq = round([px([4, 3]) * res, 3]);
    if any(size(A) > px_sq)
        A = A(1:min(end, px_sq(1)), 1:min(end, px_sq(2)), :);
    end
    if any(size(alpha) > px_sq(1:2))
        alpha = alpha(1:min(end, px_sq(1)), 1:min(end, px_sq(2)));
    end
end
end

function [imgData, alpha] = getJavaImage(hFig)
oldWarn = warning("off", "MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame");
warning("off", "MATLAB:ui:javaframe:PropertyToBeRemoved");
try
    jf = get(handle(hFig), "JavaFrame_I");
catch
    jf = get(handle(hFig), "JavaFrame");
end
warning(oldWarn);

try jClient = jf.fHG2Client; catch, try jClient = jf.fHG1Client; catch, jClient = jf.fFigureClient; end; end
try jPanel = jClient.getContentPane; catch, jPanel = jClient.getFigurePanelContainer; end
jPanel.repaint;
w = jPanel.getWidth;
h = jPanel.getHeight;

jOriginalGraphics = jPanel.getGraphics;
import java.awt.image.BufferedImage
try TYPE_INT_RGB = BufferedImage.TYPE_INT_RGB; catch, TYPE_INT_RGB = 1; end
jImage = BufferedImage(w, h, TYPE_INT_RGB);
jGraphics = jImage.createGraphics;
pause(0.05);
jPanel.paint(jGraphics);
jPanel.paint(jOriginalGraphics);
pause(0.05);

pixelsData = reshape(typecast(jImage.getData.getDataStorage, "uint8"), 4, w, h);
imgData = cat(3, transpose(reshape(pixelsData(3, :, :), w, h)), ...
    transpose(reshape(pixelsData(2, :, :), w, h)), ...
    transpose(reshape(pixelsData(1, :, :), w, h)));
alpha = transpose(reshape(pixelsData(4, :, :), w, h));
jGraphics.dispose();

figSize = getpixelposition(hFig);
if ~isequal([figSize(4), figSize(3), 3], size(imgData)), error("bad Java screen-capture size!"); end
end

function [imgData, alpha, err, ex] = getPrintImage(fig, res_str, renderer, tmp_nam)
imgData = []; err = false; ex = []; alpha = [];
fig = ancestor(fig, "figure");
old_pos_mode = get(fig, "PaperPositionMode");
old_orientation = get(fig, "PaperOrientation");
set(fig, "PaperPositionMode", "auto", "PaperOrientation", "portrait");
try
    fp = findall(fig, "Type", "patch", "LineWidth", 0.75);
    set(fp, "LineWidth", 0.5);
    try
        imgData = print(fig, renderer, res_str, "-RGBImage");
    catch
        fig_d = double(fig);
        print(fig_d, renderer, res_str, "-dtiff", char(tmp_nam));
        imgData = imread(tmp_nam);
        delete(tmp_nam);
    end
    imgSize = size(imgData); imgSize = imgSize([1, 2]);
    alpha = 255 * ones(imgSize, "uint8");
catch me
    ex = me; err = true;
end
if ~isempty(fp), set(fp, "LineWidth", 0.75); end
set(fig, "PaperPositionMode", old_pos_mode, "PaperOrientation", old_orientation);
end

function fp = fontPath()
fp = quantecon.vis.userString("gs_font_path");
if ~isempty(fp), return; end
fp = getenv("GS_FONTPATH");
if ispc
    if ~isempty(fp), fp = fp + ";"; end
    fp = fp + getenv("WINDIR") + filesep + "Fonts";
else
    if ~isempty(fp), fp = fp + ":"; end
    fp = fp + "/usr/share/fonts:/usr/local/share/fonts:/usr/share/fonts/X11:/usr/local/share/fonts/X11:/usr/share/fonts/truetype:/usr/local/share/fonts/truetype";
end
quantecon.vis.userString("gs_font_path", fp);
end
