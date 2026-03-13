function [string, file_name] = userString(string_name, string)
%USERSTRING  Get/set a user specific string
%
%   Ported to quantecon.vis.

if ~ischar(string_name) && ~isstring(string_name)
    error("string_name must be a string.");
end
% Create the full filename
fname = string_name + ".txt";
dname = fullfile(fileparts(mfilename("fullpath")), ".ignore");
file_name = fullfile(dname, fname);

if nargin > 1
    % Set string
    if ~ischar(string) && ~isstring(string)
        error("new_string must be a string.");
    end
    if ~exist(dname, "dir")
        try
            if ~mkdir(dname)
                string = false;
                return
            end
        catch
            string = false;
            return
        end
        try
            fileattrib(dname, "+h");
        catch
        end
    end
    fid = fopen(char(file_name), "w");
    if fid == -1
        string = false;
        return
    end
    fprintf(fid, "%s", string);
    fclose(fid);
else
    % Get string
    fid = fopen(char(file_name), "r");
    if fid == -1
        string = "";
        return
    end
    string = fgetl(fid);
    fclose(fid);
end
end
