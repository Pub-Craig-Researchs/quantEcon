function eps2pdf(source, dest, crop, append, gray, quality, gs_options)
%EPS2PDF  Convert an eps file to pdf format using ghostscript
%
%   Ported to quantecon.vis.

downsampleOptions = "-dDownsampleColorImages=false -dDownsampleGrayImages=false -dDownsampleMonoImages=false";
options = "-q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress " + downsampleOptions + ' -sOutputFile="' + dest + '"';

if nargin < 3 || crop, options = options + " -dEPSCrop"; end

fp = fontPath();
if ~isempty(fp), options = options + ' -sFONTPATH="' + fp + '"'; end

if nargin > 4 && gray
    options = options + " -sColorConversionStrategy=Gray -dProcessColorModel=/DeviceGray";
end

qualityOptions = "";
if nargin > 5 && ~isempty(quality)
    qualityOptions = " -dAutoFilterColorImages=false -dAutoFilterGrayImages=false";
    if quality > 100
        qualityOptions = qualityOptions + " -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode";
        qualityOptions = qualityOptions + ' -c ".setpdfwrite << /ColorImageDownsampleThreshold 10 /GrayImageDownsampleThreshold 10 >> setdistillerparams"';
    else
        qualityOptions = qualityOptions + " -dColorImageFilter=/DCTEncode -dGrayImageFilter=/DCTEncode";
        v = 1 + (quality < 80);
        q = 1 - quality / 100;
        s = sprintf("<< /QFactor %.2f /Blend 1 /HSample [%d 1 1 %d] /VSample [%d 1 1 %d] >>", q, v, v, v, v);
        qualityOptions = qualityOptions + ' -c ".setpdfwrite << /ColorImageDict ' + s + ' /GrayImageDict ' + s + ' >> setdistillerparams"';
    end
    options = options + qualityOptions;
end

if nargin > 6 && ~isempty(gs_options)
    if iscell(gs_options), gs_opt_str = sprintf(" %s", gs_options{:});
    elseif ischar(gs_options), gs_opt_str = " " + gs_options;
    else, error("gs_options must be a string or cell-array of strings"); end
    options = options + gs_opt_str;
end

if nargin > 3 && append && exist(dest, "file") == 2
    try
        file_info = dir(dest);
        orig_bytes = file_info.bytes;
    catch
        orig_bytes = [];
    end
    tmp_nam = tempname + ".pdf";
    [fpath, fname, fext] = fileparts(tmp_nam);
    try
        fid = fopen(tmp_nam, "w"); fwrite(fid, 1); fclose(fid); delete(tmp_nam);
    catch
        fpath = fileparts(dest);
        tmp_nam = fullfile(fpath, [fname, fext]);
    end
    copyfile(dest, tmp_nam);
    orig_options = options;
    quantecon.vis.ghostscript(options + ' -f "' + source + '"');
    [~, fname2] = fileparts(tempname);
    tmp_nam2 = fullfile(fpath, [fname2, fext]);
    copyfile(dest, tmp_nam2);
    options = options + ' -f "' + tmp_nam + '" "' + tmp_nam2 + '"';
    try
        [status, message] = quantecon.vis.ghostscript(options);
        if ~isempty(message) && ~isempty(orig_bytes)
            file_info = dir(dest);
            if file_info.bytes < orig_bytes + 100
                options = orig_options + ' -f "' + tmp_nam + '" "' + source + '"';
                [status, message] = quantecon.vis.ghostscript(options);
            end
        end
        delete(tmp_nam); delete(tmp_nam2);
    catch me
        delete(tmp_nam); delete(tmp_nam2); rethrow(me);
    end
else
    options = options + ' -f "' + source + '"';
    [status, message] = quantecon.vis.ghostscript(options);
end

if status
    % Error correction logic (transparency fixes, etc.)
    if ~isempty(regexpi(message, "undefined in .setopacityalpha"))
        new_options = options + " -dNOSAFER -dALLOWPSTRANSPARENCY";
        [status, message] = quantecon.vis.ghostscript(new_options);
        if ~status, return; end
        if isempty(regexpi(message, "undefined in .setopacityalpha"))
            options = new_options;
        else
            fstrm = quantecon.vis.readWriteEntireTextfile(source);
            fstrm = regexprep(fstrm, '0?\.\d+ .setopacityalpha \w+\n', "");
            quantecon.vis.readWriteEntireTextfile(source, fstrm);
            [status, message] = quantecon.vis.ghostscript(options);
            if ~status
                warning("export_fig:GS:transparency", "Alpha transparency ignored - not supported by GS.");
                return
            end
        end
    end
    error(message);
end
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
