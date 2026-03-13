function bearRoot = BearPath(opts)
%BEARPATH Return BEAR toolbox path and optionally add it to MATLAB path.
%
% Inputs:
%   opts.Root   - (string) Override BEAR root path (default: "")
%   opts.AddPath - (logical) Add BEAR root to MATLAB path (default: true)
%
% Outputs:
%   bearRoot - BEAR toolbox root path
%
% References:
%   BEAR Toolbox v5.0 (Dieppe & van Roye)

arguments
    opts.Root (1,1) string = ""
    opts.AddPath (1,1) logical = true
end

rng(0, "twister");

if opts.Root == ""
    thisFile = mfilename("fullpath");
    quanteconRoot = fileparts(fileparts(fileparts(thisFile)));
    bearRoot = fullfile(quanteconRoot, "resources", "BEAR-toolbox-master", "tbx");
else
    bearRoot = opts.Root;
end

if ~isfolder(bearRoot)
    error("quantecon:bayes:BearPath:NotFound", "BEAR toolbox not found at: %s", bearRoot);
end

if opts.AddPath
    addpath(genpath(bearRoot));
end

end
