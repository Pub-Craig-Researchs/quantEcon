function varargout = ghostscript(cmd)
%GHOSTSCRIPT  Calls a local GhostScript executable with the input command
%
%   This is a ported version for quantecon.vis.

try
    % Call ghostscript
    cmd = char(cmd);
    [varargout{1:nargout}] = system([gs_command(gs_path()) cmd]);
catch err
    % Display possible workarounds for Ghostscript croaks
    url1 = 'https://github.com/altmany/export_fig/issues/12#issuecomment-61467998';
    url2 = 'https://github.com/altmany/export_fig/issues/20#issuecomment-63826270';
    hg2_str = ''; if quantecon.vis.usingHg2(), hg2_str = ' or Matlab R2014a'; end
    fprintf(2, 'Ghostscript error. Rolling back to GS 9.10%s may possibly solve this:\n * %s ', hg2_str, quantecon.vis.hyperlink(url1));
    if quantecon.vis.usingHg2()
        fprintf(2, '(GS 9.10)\n * %s (R2014a)', quantecon.vis.hyperlink(url2));
    end
    fprintf('\n\n');
    if ismac || isunix
        url3 = 'https://github.com/altmany/export_fig/issues/27';
        fprintf(2, 'Alternatively, this may possibly be due to a font path issue:\n * %s\n\n', quantecon.vis.hyperlink(url3));
    end
    rethrow(err);
end
end

function path_ = gs_path
% Return a valid path
path_ = quantecon.vis.userString('ghostscript');
if check_gs_path(path_)
    path_ = char(path_);
    return
end
if ispc
    bin = {'gswin32c.exe', 'gswin64c.exe', 'gs', 'mgs'};
else
    bin = {'gs', 'ghostscript'};
end
for a = 1:numel(bin)
    path_ = bin{a};
    if check_store_gs_path(path_)
        path_ = char(path_);
        return
    end
end
if ispc
    default_location = 'C:\Program Files\gs\';
    dir_list = dir(default_location);
    if isempty(dir_list)
        default_location = 'C:\Program Files (x86)\gs\';
        dir_list = dir(default_location);
    end
    executable = {'\bin\gswin32c.exe', '\bin\gswin64c.exe'};
    ver_num = 0;
    for a = 1:numel(dir_list)
        ver_num2 = sscanf(dir_list(a).name, 'gs%g');
        if ~isempty(ver_num2) && ver_num2 > ver_num
            for b = 1:numel(executable)
                path2 = fullfile(default_location, dir_list(a).name, executable{b});
                if exist(path2, 'file') == 2
                    path_ = path2;
                    ver_num = ver_num2;
                end
            end
        end
    end
    if check_store_gs_path(path_)
        path_ = char(path_);
        return
    end
else
    executable = {'/usr/bin/gs', '/usr/local/bin/gs', '/opt/local/bin/gs'};
    for a = 1:numel(executable)
        path_ = executable{a};
        if check_store_gs_path(path_)
            path_ = char(path_);
            return
        end
    end
end

error('Ghostscript:NotFound', 'Ghostscript not found. Please install it.');
end

function good = check_store_gs_path(path_)
good = check_gs_path(path_);
if ~good, return; end
quantecon.vis.userString('ghostscript', char(path_));
end

function good = check_gs_path(path_)
persistent isOk
if isempty(path_)
    isOk = false;
elseif ~isequal(isOk, true)
    [status, ~] = system([gs_command(char(path_)) '-h']);
    isOk = (status == 0);
end
good = isOk;
end

function cmd = gs_command(path_)
shell_cmd = '';
if isunix, shell_cmd = 'export LD_LIBRARY_PATH=""; '; end
if ismac, shell_cmd = 'export DYLD_PATH=""; '; end
cmd = sprintf('%s"%s" ', shell_cmd, char(path_));
end
