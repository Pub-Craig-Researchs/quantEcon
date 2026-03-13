function results = BearRun(settings, opts)
%BEARRUN Execute BEARmain with prepared settings.
%
% Inputs:
%   settings      - BEAR settings object or struct
%   opts.BearRoot - (string) BEAR root path override
%   opts.AddPath  - (logical) Add BEAR to MATLAB path (default: true)
%
% Outputs:
%   results - Struct with Settings and output locations
%
% References:
%   BEAR Toolbox v5.0 (Dieppe & van Roye)

arguments
    settings (1,1) struct
    opts.BearRoot (1,1) string = ""
    opts.AddPath (1,1) logical = true
end

rng(0, "twister");

quantecon.bayes.BearPath("Root", opts.BearRoot, "AddPath", opts.AddPath);

BEARmain(settings);

results = struct();
results.Settings = settings;
if isfield(settings, "results_path")
    results.ResultsPath = settings.results_path;
end
if isfield(settings, "results_sub")
    results.ResultsSub = settings.results_sub;
end

end
