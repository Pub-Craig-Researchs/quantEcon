function varargout = pdftops(cmd)
%PDFTOPS  Calls a local pdftops executable with the input command
%
%   Ported to quantecon.vis.

if nargin < 1
    xpdf_path();
    return
end

[varargout{1:nargout}] = system([xpdf_command(xpdf_path()), cmd]);
end

function path_ = xpdf_path
path_ = quantecon.vis.userString("pdftops");
if check_xpdf_path(path_), return; end

if ispc, bin = "pdftops.exe"; else, bin = "pdftops"; end
if check_store_xpdf_path(bin), path_ = bin; return; end

if ispc, paths = {"C:\Program Files\xpdf\pdftops.exe", "C:\Program Files (x86)\xpdf\pdftops.exe"};
else, paths = {"/usr/bin/pdftops", "/usr/local/bin/pdftops"}; end

for a = 1:numel(paths)
    path_ = paths{a};
    if check_store_xpdf_path(path_), return; end
end

error("pdftops executable not found.");
end

function good = check_store_xpdf_path(path_)
good = check_xpdf_path(path_);
if ~good, return; end
quantecon.vis.userString("pdftops", path_);
end

function [good, message] = check_xpdf_path(path_)
[~, message] = system([xpdf_command(path_), "-h"]);
good = ~isempty(strfind(message, "PostScript"));
end

function cmd = xpdf_command(path_)
shell_cmd = "";
if isunix, shell_cmd = 'export LD_LIBRARY_PATH=""; '; end
if ismac, shell_cmd = 'export DYLD_LIBRARY_PATH=""; '; end
cmd = sprintf('%s"%s" ', shell_cmd, path_);
end
