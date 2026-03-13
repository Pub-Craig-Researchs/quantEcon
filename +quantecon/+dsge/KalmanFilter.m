function results = KalmanFilter(data, F, H, options)
%KALMANFILTER Linear Kalman filter with NaN (missing data) support.
%
%   State-space model:
%       z(t+1) = a + F * z(t) + eta(t),   eta(t) ~ N(0, Q)
%         y(t) = b + H * z(t) + eps(t),   eps(t) ~ N(0, R)
%
%   Missing observations (NaN in data) are automatically excluded from the
%   likelihood and update steps for that period.
%
%   INPUTS:
%       data  - (Ny x T) observation matrix. NaN = missing.
%       F     - (Nz x Nz) state transition matrix
%       H     - (Ny x Nz) measurement matrix
%
%   Name-Value Options:
%       a     - (Nz x 1) state intercept.            Default: zeros
%       b     - (Ny x 1) measurement intercept.       Default: zeros
%       Q     - (Nz x Nz) state noise covariance.     Default: eye(Nz)
%       R     - (Ny x Ny) measurement noise cov.      Default: eye(Ny)
%       G     - (Nz x Ny) cross-covariance Cov(eta,eps). Default: zeros
%       z0    - (Nz x 1) initial state mean.          Default: auto
%       P0    - (Nz x Nz) initial state covariance.   Default: auto
%       Lead  - (scalar) forecast steps beyond T.     Default: 0
%
%   OUTPUTS (struct):
%       results.LogLik      - total log-likelihood
%       results.zEnd        - (Nz x 1) final filtered state
%       results.PEnd        - (Nz x Nz) final filtered covariance
%       results.Pred        - (Nz x T+Lead) predicted states
%       results.VPred       - (Nz x Nz x T+Lead) predicted covariances
%       results.Filt        - (Nz x T) filtered states
%       results.VFilt       - (Nz x Nz x T) filtered covariances
%       results.ForecastErr - (Ny x T) forecast errors (NaN where missing)
%
%   Reference:
%       Durbin, J. & Koopman, S.J. (2012). "Time Series Analysis by State
%       Space Methods." 2nd ed., Oxford University Press.
%       Originally adapted from Federal Reserve Bank of Atlanta code
%       (Karibzhanov, 2002).
%
%   See also: quantecon.dsge.KalmanSmoother, quantecon.dsge.Gensys

arguments
    data  (:,:) double
    F     (:,:) double
    H     (:,:) double
    options.a  (:,1) double = []
    options.b  (:,1) double = []
    options.Q  (:,:) double = []
    options.R  (:,:) double = []
    options.G  (:,:) double = []
    options.z0 (:,1) double = []
    options.P0 (:,:) double = []
    options.Lead (1,1) double {mustBeNonnegative, mustBeInteger} = 0
end

[Ny, T] = size(data);
Nz = size(F, 1);

% --- Defaults ---
a = options.a;  if isempty(a); a = zeros(Nz, 1); end
b = options.b;  if isempty(b); b = zeros(Ny, 1); end
Q = options.Q;  if isempty(Q); Q = eye(Nz);      end
R = options.R;  if isempty(R); R = eye(Ny);       end
G = options.G;  if isempty(G); G = zeros(Nz, Ny); end
lead = options.Lead;

% --- Dimension checks ---
assert(size(F,2) == Nz, 'F must be Nz x Nz');
assert(all(size(H) == [Ny, Nz]), 'H must be Ny x Nz');
assert(all(size(Q) == [Nz, Nz]), 'Q must be Nz x Nz');
assert(all(size(R) == [Ny, Ny]), 'R must be Ny x Ny');

% --- Initial conditions ---
if ~isempty(options.z0)
    z = options.z0;
    P = options.P0;
else
    % Try stationary initialisation; fallback to diffuse
    ev = eig(F);
    if all(abs(ev) < 1 - 1e-10)
        z = (eye(Nz) - F) \ a;
        P = reshape((eye(Nz^2) - kron(F, F)) \ Q(:), Nz, Nz);
    else
        z = a;
        P = eye(Nz) * 1e6;
    end
end

% --- Preallocate ---
pred   = zeros(Nz, T + lead);
vpred  = zeros(Nz, Nz, T + lead);
filt   = zeros(Nz, T);
vfilt  = zeros(Nz, Nz, T);
ferr   = NaN(Ny, T);
loglik = 0;

% --- Forward pass ---
for t = 1:T
    % Prediction step
    z = a + F * z;
    P = F * P * F' + Q;

    % Handle missing data
    obs = ~isnan(data(:, t));
    y_t = data(obs, t);
    H_t = H(obs, :);
    b_t = b(obs);
    R_t = R(obs, obs);
    G_t = G(:, obs);
    Ny_t = sum(obs);

    % Store prediction
    pred(:, t) = z;
    vpred(:, :, t) = P;

    if Ny_t == 0
        % No observations: skip update
        filt(:, t) = z;
        vfilt(:, :, t) = P;
        continue
    end

    % Innovation
    dy = y_t - H_t * z - b_t;
    ferr(obs, t) = dy;

    % Innovation covariance
    HG = H_t * G_t;
    S = H_t * P * H_t' + HG + HG' + R_t;
    S = 0.5 * (S + S');  % enforce symmetry

    % Log-likelihood increment
    % [FIX]: use S\ instead of inv(S) for numerical stability
    S_dy = S \ dy;
    loglik = loglik - 0.5 * (log(det(S)) + dy' * S_dy + Ny_t * log(2*pi));

    % Update step
    K = (P * H_t' + G_t);  % Kalman gain numerator
    z = z + K * S_dy;
    P = P - K / S * K';
    P = 0.5 * (P + P');  % enforce symmetry

    filt(:, t) = z;
    vfilt(:, :, t) = P;
end

% --- Forecast beyond data ---
for t = T+1:T+lead
    z = F * z + a;
    P = F * P * F' + Q;
    pred(:, t) = z;
    vpred(:, :, t) = P;
end

% --- Pack output ---
results.LogLik      = loglik;
results.zEnd        = z;
results.PEnd        = P;
results.Pred        = pred;
results.VPred       = vpred;
results.Filt        = filt;
results.VFilt       = vfilt;
results.ForecastErr = ferr;

end
