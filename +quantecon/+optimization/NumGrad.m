function [g, badg] = NumGrad(fcn, x, varargin)
%NUMGRAD Forward-difference numerical gradient
%
%   Computes the gradient of a scalar-valued function using one-sided
%   forward finite differences.
%
%   Reference:
%       Sims, C.A. — companion to csminwel.m
%
%   Usage:
%       [g, badg] = quantecon.optimization.NumGrad(@fcn, x0);
%       [g, badg] = quantecon.optimization.NumGrad(@fcn, x0, p1, p2);
%
%   Inputs:
%       fcn  - function handle:  f = fcn(x, varargin{:})
%       x    - (n x 1) evaluation point
%       varargin - extra arguments forwarded to fcn
%
%   Outputs:
%       g    - (n x 1) gradient vector
%       badg - (logical) true if any component appears unreliable (|g_i| >= 1e15)

% [FIX]: All eval() replaced with direct function-handle calls

if ~isa(fcn, 'function_handle')
    error('quantecon:optimization:NumGrad:badFcn', ...
          'fcn must be a function handle.');
end

x = x(:);
n = length(x);
delta = 1e-6;

g    = zeros(n, 1);
badg = false;

f0 = fcn(x, varargin{:});

for i = 1:n
    ei    = zeros(n, 1);
    ei(i) = delta;
    g0 = (fcn(x + ei, varargin{:}) - f0) / delta;
    if abs(g0) < 1e15
        g(i) = g0;
    else
        g(i)  = 0;
        badg  = true;
    end
end
end
