function fh = copyFig(fh)
%COPYFIG Create a copy of a figure
%
%   Ported to quantecon.vis.

if nargin == 0, fh = gcf; end

% Is there a legend? (Legend handles are special in older Matlab)
useCopyobj = isempty(findall(fh, "Type", "axes", "Tag", "legend"));

if useCopyobj
    oldWarn = warning("off");
    try
        fh = copyobj(fh, 0);
    catch
        useCopyobj = false;
    end
    warning(oldWarn);
end

if ~useCopyobj
    tmp_nam = tempname + ".fig";
    try
        fid = fopen(tmp_nam, "w"); fwrite(fid, 1); fclose(fid); delete(tmp_nam);
    catch
        tmp_nam = fullfile(pwd, [char(tempname), ".fig"]);
    end
    hgsave(fh, tmp_nam);
    fh = hgload(tmp_nam);
    delete(tmp_nam);
end
end
