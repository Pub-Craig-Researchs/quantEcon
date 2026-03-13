function [G1, C, impact, fmat, fwt, ywt, gev, eu] = Gensys(g0, g1, c, psi, pi, options)
%GENSYS Solve linear rational expectations models via QZ decomposition.
%
%   Solves the system:
%       g0 * y(t) = g1 * y(t-1) + c + psi * z(t) + pi * eta(t)
%
%   Returns the solution:
%       y(t) = G1 * y(t-1) + C + impact * z(t)
%            + ywt * inv(I - fmat * inv(L)) * fwt * z(t+1)
%
%   If z(t) is i.i.d., the last term drops out.
%
%   INPUTS:
%       g0    - (n x n) coefficient matrix on current states
%       g1    - (n x n) coefficient matrix on lagged states
%       c     - (n x 1) constant vector
%       psi   - (n x nz) coefficient matrix on exogenous shocks z(t)
%       pi    - (n x npi) coefficient matrix on expectational errors eta(t)
%       Div   - (optional) threshold for stable/unstable root separation.
%               Default: auto-calculated slightly above 1.
%
%   OUTPUTS:
%       G1     - (n x n)   transition matrix
%       C      - (n x 1)   constant
%       impact - (n x nz)  shock impact matrix
%       fmat   - (nu x nu) unstable block dynamics
%       fwt    - (nu x nz) unstable block shock weights
%       ywt    - (n x nu)  link from unstable block to full state
%       gev    - (n x 2)   generalized eigenvalues [diag(a), diag(b)]
%       eu     - (2 x 1)   existence/uniqueness flags:
%                           eu(1)=1 existence, eu(2)=1 uniqueness
%                           eu=[-2;-2] coincident zeros
%                           eu=[-4;-4] numerical failure (Inf/NaN or QZ error)
%
%   Reference:
%       Sims, C.A. (2002). "Solving Linear Rational Expectations Models."
%       Computational Economics, 20(1-2), 1-20.
%
%   Original code by Christopher A. Sims.
%   Updated by Marco Del Negro, Vasco Curdia, Daria Finocchiaro.
%   Refactored for +quantecon by removing eval(), replacing inv() with \.
%
%   See also: quantecon.dsge.PertSolve, quantecon.dsge.KalmanFilter

arguments
    g0   (:,:) double
    g1   (:,:) double
    c    (:,1) double
    psi  (:,:) double
    pi   (:,:) double
    options.Div (1,1) double {mustBeNonnegative} = 0
end

n = size(g0, 1);
div = options.Div;
fixdiv = (div > 0);

% --- Initialise outputs for early return ---
eu = [0; 0];
realsmall = 1e-6;
empty_out = @() deal([], [], [], [], [], [], [], [-4; -4]);

% --- Validate inputs ---
if any(~isfinite(g0), 'all') || any(~isfinite(g1), 'all')
    [G1, C, impact, fmat, fwt, ywt, gev, eu] = empty_out();
    return
end

% --- QZ decomposition ---
try
    [a, b, q, z] = qz(g0, g1);
catch
    [G1, C, impact, fmat, fwt, ywt, gev, eu] = empty_out();
    return
end

if ~fixdiv
    div = 1.01;
end

% --- Count unstable roots and coincident zeros ---
nunstab = 0;
zxz = false;
for i = 1:n
    if ~fixdiv
        if abs(a(i,i)) > 0
            divhat = abs(b(i,i)) / abs(a(i,i));
            if (1 + realsmall) < divhat && divhat < div
                div = 0.5 * (1 + divhat);
            end
        end
    end
    nunstab = nunstab + (abs(b(i,i)) > div * abs(a(i,i)));
    if abs(a(i,i)) < realsmall && abs(b(i,i)) < realsmall
        zxz = true;
    end
end

% --- Reorder QZ so unstable roots are in lower-right ---
if ~zxz
    [a, b, q, z] = qzdiv_local(div, a, b, q, z);
end

gev = [diag(a), diag(b)];

if zxz
    eu = [-2; -2];
    G1 = []; C = []; impact = []; fmat = []; fwt = []; ywt = [];
    return
end

% --- Partition ---
q1 = q(1:n-nunstab, :);
q2 = q(n-nunstab+1:n, :);
z1 = z(:, 1:n-nunstab)';
z2 = z(:, n-nunstab+1:n)';
a2 = a(n-nunstab+1:n, n-nunstab+1:n);
b2 = b(n-nunstab+1:n, n-nunstab+1:n);
etawt = q2 * pi;
zwt   = q2 * psi;

[ueta, deta, veta] = svd(etawt);
md = min(size(deta));
bigev = find(diag(deta(1:md, 1:md)) > realsmall);
ueta = ueta(:, bigev);
veta = veta(:, bigev);
deta = deta(bigev, bigev);

[uz, dz, vz] = svd(zwt);
md = min(size(dz));
bigev = find(diag(dz(1:md, 1:md)) > realsmall);
uz = uz(:, bigev);
dz = dz(bigev, bigev);

if isempty(bigev)
    existflag = true;
else
    existflag = norm(uz - ueta * (ueta' * uz)) < realsmall * n;
end

if ~isempty(bigev)
    % [FIX]: replaced inv(b2) with b2\ for numerical stability
    zwtx0 = b2 \ zwt;
    zwtx = zwtx0;
    M = b2 \ a2;
    for i = 2:nunstab
        zwtx = [M * zwtx, zwtx0]; %#ok<AGROW>
    end
    zwtx = b2 * zwtx;

    if any(~isfinite(zwtx), 'all')
        [G1, C, impact, fmat, fwt, ywt, gev, eu] = empty_out();
        return
    end

    [ux, dx, ~] = svd(zwtx);
    md = min(size(dx));
    bigev = find(diag(dx(1:md, 1:md)) > realsmall);
    ux = ux(:, bigev);
    existx = norm(ux - ueta * (ueta' * ux)) < realsmall * n;
else
    existx = true;
end

% --- Existence check ---
[ueta1, deta1, veta1] = svd(q1 * pi);
md = min(size(deta1));
bigev = find(diag(deta1(1:md, 1:md)) > realsmall);
ueta1 = ueta1(:, bigev);
veta1 = veta1(:, bigev);
deta1 = deta1(bigev, bigev);

if existx || nunstab == 0
    eu(1) = 1;
else
    if existflag
        eu(1) = -1;
    end
end

% --- Uniqueness check ---
if isempty(veta1)
    unique_flag = true;
else
    unique_flag = norm(veta1 - veta * (veta' * veta1)) < realsmall * n;
end

if unique_flag
    eu(2) = 1;
end

% --- Build solution ---
tmat = [eye(n - nunstab), -(ueta * (deta \ veta') * veta1 * deta1 * ueta1')'];
G0_sol = [tmat * a; zeros(nunstab, n - nunstab), eye(nunstab)];
G1_sol = [tmat * b; zeros(nunstab, n)];

% [FIX]: replaced inv(G0_sol) with G0_sol\ for numerical stability
G0I = G0_sol \ eye(n);
G1_sol = G0I * G1_sol;
usix = n - nunstab + 1:n;
C = G0I * [tmat * q * c; (a(usix, usix) - b(usix, usix)) \ (q2 * c)];
impact = G0I * [tmat * q * psi; zeros(nunstab, size(psi, 2))];
fmat = b(usix, usix) \ a(usix, usix);
fwt  = -(b(usix, usix) \ (q2 * psi));
ywt  = G0I(:, usix);

% --- Transform back to original coordinates ---
G1     = real(z * G1_sol * z');
C      = real(z * C);
impact = real(z * impact);
ywt    = z * ywt;

end

% =========================================================================
% LOCAL: qzdiv - reorder QZ decomposition
% =========================================================================
function [A, B, Q, Z] = qzdiv_local(stake, A, B, Q, Z)
    n = size(A, 1);
    root = abs([diag(A), diag(B)]);
    root(:,1) = root(:,1) - (root(:,1) < 1e-13) .* (root(:,1) + root(:,2));
    root(:,2) = root(:,2) ./ root(:,1);
    for i = n:-1:1
        m = 0;
        for j = i:-1:1
            if root(j,2) > stake || root(j,2) < -0.1
                m = j;
                break
            end
        end
        if m == 0
            return
        end
        for k = m:i-1
            [A, B, Q, Z] = qzswitch_local(k, A, B, Q, Z);
            tmp = root(k, 2);
            root(k, 2) = root(k+1, 2);
            root(k+1, 2) = tmp;
        end
    end
end

% =========================================================================
% LOCAL: qzswitch - swap diagonal elements i and i+1
% =========================================================================
function [A, B, Q, Z] = qzswitch_local(i, A, B, Q, Z)
    realsmall = sqrt(eps) * 10;
    a = A(i,i); d = B(i,i); b = A(i,i+1); e = B(i,i+1);
    cc = A(i+1,i+1); f = B(i+1,i+1);

    if abs(cc) < realsmall && abs(f) < realsmall
        if abs(a) < realsmall
            return
        else
            wz = [b; -a];
            wz = wz / norm(wz);
            wz = [wz, [wz(2); -wz(1)]];
            xy = eye(2);
        end
    elseif abs(a) < realsmall && abs(d) < realsmall
        if abs(cc) < realsmall
            return
        else
            wz = eye(2);
            xy = [cc, -b];
            xy = xy / norm(xy);
            xy = [[xy(2), -xy(1)]; xy];
        end
    else
        wz = [cc*e - f*b, (cc*d - f*a)'];
        xy = [(b*d - e*a)', (cc*d - f*a)'];
        nn = norm(wz);
        mm = norm(xy);
        if mm < eps * 100
            return
        end
        wz = wz / nn;
        xy = xy / mm;
        wz = [wz; -wz(2)', wz(1)'];
        xy = [xy; -xy(2)', xy(1)'];
    end
    A(i:i+1, :) = xy * A(i:i+1, :);
    B(i:i+1, :) = xy * B(i:i+1, :);
    A(:, i:i+1) = A(:, i:i+1) * wz;
    B(:, i:i+1) = B(:, i:i+1) * wz;
    Z(:, i:i+1) = Z(:, i:i+1) * wz;
    Q(i:i+1, :) = xy * Q(i:i+1, :);
end
