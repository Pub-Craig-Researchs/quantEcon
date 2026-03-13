function tf = usingHg2(fig)
%USINGHG2 Determine if the HG2 graphics engine is used
%
%   Ported to quantecon.vis.

persistent tf_cached
if isempty(tf_cached)
    try
        if nargin < 1, fig = figure("visible", "off"); end
        oldWarn = warning("off", "MATLAB:graphicsversion:GraphicsVersionRemoval");
        try
            tf = ~graphicsversion(fig, "handlegraphics");
        catch
            tf = ~verLessThan("matlab", "8.4");
        end
        warning(oldWarn);
    catch
        tf = false;
    end
    if nargin < 1, delete(fig); end
    tf_cached = tf;
else
    tf = tf_cached;
end
end
