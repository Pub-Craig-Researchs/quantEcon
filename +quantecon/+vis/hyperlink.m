function str = hyperlink(url, label, msg)
%HYPERLINK  create a string that is displayable as hyperlink in Matlab console
%
%   Ported to quantecon.vis.

if nargin < 1, error("Not enough input arguments."); end
if nargin > 2 % msg was specified
    str = strrep(msg, label, quantecon.vis.hyperlink(url, label));
    return
end
if nargin < 2, label = url; end
isWebpage = strncmpi(url, "http", 4);

if ~isdeployed
    if isWebpage
        str = sprintf('<a href="matlab:web(''-browser'',''%s'');">%s</a>', url, label);
    else
        str = sprintf('<a href="%s">%s</a>', url, label);
    end
else
    if isWebpage && ~strcmp(label, url)
        str = label + " (" + url + ")";
    elseif isWebpage
        str = url;
    else
        str = label;
    end
end
end
