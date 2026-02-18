function result = Kde(data, opts)
% Kde  Kernel density estimation via diffusion (Botev et al. 2010)
%
%   Automatic bandwidth selection, immune to multimodal densities.
%   Gaussian kernel assumed. Much faster than ksdensity for large N.
%
% SYNTAX:
%   result = quantecon.base.Kde(data)
%   result = quantecon.base.Kde(data, NGrid=2^14, Min=-5, Max=5)
%
% INPUT:
%   data - (T x 1) vector of observations
%
% OPTIONS:
%   NGrid - number of grid points (rounded up to next power of 2)
%           (default 2^14)
%   Min   - lower bound of estimation interval  (default auto)
%   Max   - upper bound of estimation interval  (default auto)
%
% OUTPUT:
%   result struct with fields:
%     .bandwidth - optimal bandwidth (Gaussian kernel)
%     .density   - (NGrid x 1) density values on grid
%     .xmesh     - (1 x NGrid) evaluation grid
%     .cdf       - (NGrid x 1) cumulative distribution values
%
% REFERENCE:
%   Botev, Z.I., Grotowski, J.F. and Kroese, D.P. (2010) "Kernel
%   density estimation via diffusion", Annals of Statistics, 38(5),
%   2916-2957.
%
% Source: Refactored from Botev kde.m (MATLAB File Exchange).

arguments
    data (:,1) double
    opts.NGrid (1,1) double {mustBePositive} = 2^14
    opts.Min (1,1) double = NaN
    opts.Max (1,1) double = NaN
end

data = data(:);
n = 2^ceil(log2(opts.NGrid));

% Default interval
mn = min(data); mx = max(data);
R0 = mx - mn;
if isnan(opts.Min), MIN = mn - R0/2; else, MIN = opts.Min; end
if isnan(opts.Max), MAX = mx + R0/2; else, MAX = opts.Max; end

R  = MAX - MIN;
dx = R / (n - 1);
xmesh = MIN + (0:dx:R);
N = length(unique(data));

% Bin data
initial_data = histc(data, xmesh) / N; %#ok<HISTC>
initial_data = initial_data / sum(initial_data);

% Discrete cosine transform
a = dct1d(initial_data);

% Solve for optimal bandwidth
I  = (1:n-1)'.^2;
a2 = (a(2:end) / 2).^2;
t_star = find_root(@(t) fixed_point(t, N, I, a2), N);

% Smooth and inverse DCT
a_t = a .* exp(-(0:n-1)'.^2 * pi^2 * t_star / 2);
density = idct1d(a_t) / R;
density(density < 0) = eps;  % [FIX]: remove negatives from round-off

bandwidth = sqrt(t_star) * R;

% CDF estimation
f_cdf = 2*pi^2 * sum(I .* a2 .* exp(-I * pi^2 * t_star));
t_cdf = (sqrt(pi) * f_cdf * N)^(-2/3);
a_cdf = a .* exp(-(0:n-1)'.^2 * pi^2 * t_cdf / 2);
cdf_vals = cumsum(idct1d(a_cdf)) * (dx / R);

result = struct('bandwidth', bandwidth, 'density', density, ...
                'xmesh', xmesh, 'cdf', cdf_vals);
end

% =====================================================================

function out = fixed_point(t, N, I, a2)
% Implements t - zeta * gamma^[l](t)
    ell = 7;
    f = 2 * pi^(2*ell) * sum(I.^ell .* a2 .* exp(-I * pi^2 * t));
    for s = ell-1:-1:2
        K0 = prod(1:2:2*s-1) / sqrt(2*pi);
        c  = (1 + (1/2)^(s + 1/2)) / 3;
        time = (2*c*K0 / N / f)^(2/(3+2*s));
        f = 2 * pi^(2*s) * sum(I.^s .* a2 .* exp(-I * pi^2 * time));
    end
    out = t - (2*N*sqrt(pi)*f)^(-2/5);
end

function data = dct1d(data)
% Discrete cosine transform (Type II)
    nrows = size(data, 1);
    weight = [1; 2*(exp(-1i*(1:nrows-1)*pi/(2*nrows))).'];
    data = [data(1:2:end,:); data(end:-2:2,:)];
    data = real(weight .* fft(data));
end

function out = idct1d(data)
% Inverse discrete cosine transform
    nrows = size(data, 1);
    weights = nrows * exp(1i*(0:nrows-1)*pi/(2*nrows)).';
    data = real(ifft(weights .* data));
    out = zeros(nrows, 1);
    out(1:2:nrows) = data(1:nrows/2);
    out(2:2:nrows) = data(nrows:-1:nrows/2+1);
end

function t = find_root(f, N)
% Find smallest root via fzero with expanding tolerance
    N = max(50, min(1050, N));
    tol = 1e-12 + 0.01*(N - 50)/1000;
    flag = false;
    while ~flag
        try
            t = fzero(f, [0, tol]);
            flag = true;
        catch
            tol = min(tol*2, 0.1);
        end
        if tol >= 0.1
            t = fminbnd(@(x) abs(f(x)), 0, 0.1);
            flag = true;
        end
    end
end
