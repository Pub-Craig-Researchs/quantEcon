function varargout = exportFig(varargin)
%EXPORTFIG  Exports a figure to a high-quality image or vector file
%
%   Ported to quantecon.vis.

drawnow; pause(0.05);

% Default figure handle
fig = gcf();

% Parse arguments
[fig, options] = parseArgs(fig, varargin{:});

if isempty(fig) || ~ishandle(fig), error('Invalid figure handle.'); end

% Check for Ghostscript
hasGS = false;
try
    quantecon.vis.ghostscript('-v');
    hasGS = true;
catch
    % Ghostscript not found - will use exportgraphics as fallback for PDF/EPS
end

% Handle transparency if requested
if options.transparent
    tcol_orig = get(fig, 'Color');
    set(fig, 'Color', 'none');
end

% Logic for bitmap vs vector
is_vector = options.pdf || options.eps;
is_bitmap = options.png || options.tif || options.jpg || options.bmp;

% VECTOR EXPORT
if is_vector
    if hasGS
        % Traditional export_fig logic
        eps_name = [char(tempname), '.eps'];
        quantecon.vis.print2Eps(eps_name, fig, options);

        if options.pdf
            pdf_name = [char(options.name), '.pdf'];
            quantecon.vis.eps2pdf(eps_name, pdf_name, options.crop, options.append);
        end

        if options.eps
            eps_final = [char(options.name), '.eps'];
            movefile(eps_name, eps_final, 'f');
        else
            if exist(eps_name, 'file'), delete(eps_name); end
        end
    else
        % Fallback to exportgraphics (Matlab R2020a+)
        if options.pdf
            filename = [char(options.name), '.pdf'];
            exportgraphics(fig, filename, 'ContentType', 'vector', ...
                'BackgroundColor', 'none', 'Append', options.append);
        end
        if options.eps
            filename = [char(options.name), '.eps'];
            exportgraphics(fig, filename, 'ContentType', 'vector', ...
                'BackgroundColor', 'none');
        end
    end
end

% BITMAP EXPORT
if is_bitmap
    try
        [A, bcol, alpha] = quantecon.vis.print2Array(fig, options.magnify, options.renderer);

        if options.crop
            A = quantecon.vis.cropBorders(A, bcol);
            if ~isempty(alpha), alpha = quantecon.vis.cropBorders(alpha, 255); end
        end

        if options.png
            filename = [char(options.name), '.png'];
            if options.transparent && ~isempty(alpha)
                imwrite(A, filename, 'Alpha', double(alpha)/255);
            else
                imwrite(A, filename);
            end
        end

        if options.jpg
            imwrite(A, [char(options.name), '.jpg'], 'Quality', 95);
        end

        if nargout > 0
            varargout{1} = A;
            if nargout > 1, varargout{2} = alpha; end
        end
    catch
        % Fallback to exportgraphics for bitmap if print2Array fails
        res = round(options.magnify * 72);
        if options.png
            exportgraphics(fig, [char(options.name), '.png'], 'Resolution', res);
        end
    end
end

if options.transparent
    set(fig, 'Color', tcol_orig);
end
end

function [fig, options] = parseArgs(fig, varargin)
options = struct('name', 'export_fig_out', 'pdf', false, 'eps', false, ...
    'png', false, 'tif', false, 'jpg', false, 'bmp', false, ...
    'transparent', false, 'magnify', 1, 'renderer', '-opengl', ...
    'crop', true, 'append', false, 'quality', 95);

found_format = false;

for i = 1:numel(varargin)
    arg = varargin{i};
    if ishandle(arg) && any(strcmpi(get(arg, 'Type'), {'figure', 'axes'}))
        fig = arg;
    elseif ischar(arg) || isa(arg, 'string')
        arg = char(arg);
        if ~isempty(arg) && arg(1) == '-'
            opt = lower(arg(2:end));
            if startsWith(opt, 'm')
                options.magnify = str2double(opt(2:end));
            elseif startsWith(opt, 'r')
                options.magnify = str2double(opt(2:end)) / 72;
            elseif strcmp(opt, 'pdf'), options.pdf = true; found_format = true;
            elseif strcmp(opt, 'eps'), options.eps = true; found_format = true;
            elseif strcmp(opt, 'png'), options.png = true; found_format = true;
            elseif strcmp(opt, 'jpg'), options.jpg = true; found_format = true;
            elseif strcmp(opt, 'tif'), options.tif = true; found_format = true;
            elseif strcmp(opt, 'transparent'), options.transparent = true;
            elseif strcmp(opt, 'nocrop'), options.crop = false;
            elseif strcmp(opt, 'append'), options.append = true;
            elseif strcmp(opt, 'painters'), options.renderer = '-painters';
            elseif strcmp(opt, 'opengl'), options.renderer = '-opengl';
            end
        else
            options.name = arg;
        end
    end
end

if ~found_format && ~nargout
    options.png = true;
end
end
