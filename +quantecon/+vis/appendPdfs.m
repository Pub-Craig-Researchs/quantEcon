function appendPdfs(varargin)
%APPENDPDFS Appends/concatenates multiple PDF files
%
%   Ported to quantecon.vis.

if nargin < 2, return; end

% Convert strings to chars and trim
for i = 1:nargin
    if isa(varargin{i}, "string"), varargin{i} = char(varargin{i}); end
    varargin{i} = strtrim(varargin{i});
end

if nargin == 2 && iscell(varargin{2})
    varargin = {varargin{1}, varargin{2}{:}};
end

numArgs = numel(varargin);
if numArgs < 2, error("appendPdfs: Missing input filenames"); end

append = ~isempty(dir(varargin{1}));
if ~append && numArgs == 2
    copyfile(varargin{2}, varargin{1});
    return
end

output = tempname + ".pdf";
try
    fid = fopen(output, "w"); fwrite(fid, 1); fclose(fid); delete(output);
    isTempDirOk = true;
catch
    [~, fname, fext] = fileparts(output);
    fpath = fileparts(varargin{1});
    output = fullfile(fpath, [fname, fext]);
    isTempDirOk = false;
end

if ~append
    output = varargin{1};
    varargin = varargin(2:end);
end

for fileIdx = 1:numel(varargin)
    filename = char(varargin{fileIdx});
    [~, ~, ext] = fileparts(filename);
    if isempty(ext) || isempty(dir(filename))
        filename2 = filename + ".pdf";
        if ~isempty(dir(filename2)), varargin{fileIdx} = filename2;
        else, error("appendPdfs: Input file %s does not exist", filename); end
    end
end

if isTempDirOk, cmdfile = tempname + ".txt"; else, cmdfile = fullfile(fpath, [fname, ".txt"]); end
prepareCmdFile(cmdfile, output, varargin{:});
hCleanup = onCleanup(@() cleanup(cmdfile));

[status, errMsg] = quantecon.vis.ghostscript("@" + '"' + cmdfile + '"');

if status
    error("appendPdfs:ghostscriptError", errMsg);
end

if append, movefile(output, varargin{1}, "f"); end
end

function cleanup(cmdfile)
try delete(cmdfile); catch, end
end

function prepareCmdFile(cmdfile, output, varargin)
if ispc, output(output == "\") = "/"; varargin = strrep(varargin, "\", "/"); end
varargin = strrep(varargin, '"', "");

str = "-q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress " + ...
    '-sOutputFile="' + output + '" -f ' + sprintf('"%s" ', varargin{:});
str = regexprep(str, ' "?" ', " ");
str = regexprep(str, '"([^ ]*)"', "$1");
str = strtrim(str);

fh = fopen(cmdfile, "w");
fprintf(fh, "%s", str);
fclose(fh);
end
