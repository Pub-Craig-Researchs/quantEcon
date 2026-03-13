function settings = BearSettings(varType, opts)
%BEARSETTINGS Create BEAR settings object with quantecon defaults.
%
% Inputs:
%   varType       - BEAR VAR type (numeric or string)
%   opts.BearRoot - (string) BEAR root path override
%   opts.AddPath  - (logical) Add BEAR to MATLAB path (default: true)
%   opts.ExcelFile - (string) Excel data file (optional)
%   opts.Extra    - (cell) Additional name-value pairs for BEARsettings
%
% Outputs:
%   settings - BEAR settings object
%
% References:
%   BEAR Toolbox v5.0 (Dieppe & van Roye)

arguments
    varType {mustBeNumericOrString}
    opts.BearRoot (1,1) string = ""
    opts.AddPath (1,1) logical = true
    opts.ExcelFile (1,1) string = ""
    opts.Extra (1,:) cell = {}
end

rng(0, "twister");

quantecon.bayes.BearPath("Root", opts.BearRoot, "AddPath", opts.AddPath);

if opts.ExcelFile == ""
    settings = BEARsettings(varType, opts.Extra{:});
else
    settings = BEARsettings(varType, "ExcelFile", opts.ExcelFile, opts.Extra{:});
end

end
