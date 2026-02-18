function pdf2Eps(source, dest)
%PDF2EPS  Convert a pdf file to eps format using pdftops
%
%   Ported to quantecon.vis.

options = "-q -paper match -eps -level2 " + '"' + source + '" "' + dest + '"';
[status, message] = quantecon.vis.pdftops(options);

if status
    if isempty(message)
        error("Unable to generate eps. Check destination directory is writable.");
    else
        error(message);
    end
end

% Fix the DSC error created by pdftops
fid = fopen(dest, "r+");
if fid == -1, return; end
fgetl(fid);
str = fgetl(fid);
if strcmp(str(1:min(13, end)), "% Produced by")
    fseek(fid, -numel(str)-1, "cof");
    fwrite(fid, "%");
end
fclose(fid);
end
