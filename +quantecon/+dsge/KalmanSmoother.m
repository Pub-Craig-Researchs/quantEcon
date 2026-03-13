function results = KalmanSmoother(data, F, H, R_shock, Q_noise, options)
%KALMANSMOOTHER Koopman (1993) disturbance smoother for state-space models.
%
%   Implements the fixed-interval smoother from Koopman (1993) /
%   Durbin & Koopman (2002) which avoids large matrix inversions.
%
%   State-space model (no measurement error by default):
%       z(t+1) = T * z(t) + R * eta(t+1),   eta ~ N(0, Q)
%         y(t) = Z * z(t) + b
%
%   INPUTS:
%       data    - (Ny x T) observation matrix. NaN = missing.
%       F       - (Nz x Nz) state transition matrix (T in DK notation)
%       H       - (Ny x Nz) measurement matrix (Z in DK notation)
%       R_shock - (Nz x Ne) shock impact matrix
%       Q_noise - (Ne x Ne) shock covariance matrix
%
%   Name-Value Options:
%       b       - (Ny x 1) measurement intercept.    Default: zeros
%       z0      - (Nz x 1) initial state mean.       Default: zeros
%       P0      - (Nz x Nz) initial state covariance. Default: dlyap or 1e6*I
%
%   OUTPUTS (struct):
%       results.SmoothedStates - (Nz x T) smoothed state estimates
%       results.SmoothedShocks - (Ne x T) smoothed shock estimates
%       results.Pred           - (Nz x T) one-step predicted states (from filter)
%       results.VPred          - (Nz x Nz x T) predicted covariances
%
%   Reference:
%       Koopman, S.J. (1993). "Disturbance Smoother for State Space Models."
%       Biometrika, 80(1), 117-126.
%       Durbin, J. & Koopman, S.J. (2002). "A Simple and Efficient
%       Simulation Smoother." Biometrika, 89(3), 603-616.
%
%   See also: quantecon.dsge.KalmanFilter

arguments
    data    (:,:) double
    F       (:,:) double
    H       (:,:) double
    R_shock (:,:) double
    Q_noise (:,:) double
    options.b  (:,1) double = []
    options.z0 (:,1) double = []
    options.P0 (:,:) double = []
end

[Ny, T] = size(data);
Nz = size(F, 1);
Ne = size(R_shock, 2);

b = options.b;  if isempty(b); b = zeros(Ny, 1); end

% --- Initial conditions ---
if ~isempty(options.z0)
    z0 = options.z0;
    P0 = options.P0;
else
    z0 = zeros(Nz, 1);
    try
        P0 = dlyap(F, R_shock * Q_noise * R_shock');
    catch
        P0 = eye(Nz) * 1e6;
    end
end

% ========================
% Forward pass: Kalman filter
% ========================
pred  = zeros(Nz, T);
vpred = zeros(Nz, Nz, T);

z = z0;
P = P0;
V_state = R_shock * Q_noise * R_shock';

for t = 1:T
    % Prediction
    z = F * z;
    P = F * P * F' + V_state;
    pred(:, t) = z;
    vpred(:, :, t) = P;

    % Handle missing data
    obs = ~isnan(data(:, t));
    if ~any(obs)
        continue
    end
    y_t = data(obs, t);
    Z_t = H(obs, :);
    b_t = b(obs);

    % Innovation
    v = y_t - Z_t * z - b_t;
    S = Z_t * P * Z_t';
    S = 0.5 * (S + S');

    % Update
    % [FIX]: use S\ instead of inv(S) for stability
    K = F * P * Z_t' / S;
    z = z + (P * Z_t') * (S \ v);
    P = P - (P * Z_t') / S * (Z_t * P);
    P = 0.5 * (P + P');
end

% ========================
% Backward pass: disturbance smoother
% ========================
r = zeros(Nz, T);
r_t = zeros(Nz, 1);
eta_hat = zeros(Ne, T);

for t = T:-1:1
    obs = ~isnan(data(:, t));
    if ~any(obs)
        r(:, t) = r_t;
        continue
    end
    y_t = data(obs, t);
    Z_t = H(obs, :);
    b_t = b(obs);

    a_t = pred(:, t);
    P_t = vpred(:, :, t);

    S = Z_t * P_t * Z_t';
    S = 0.5 * (S + S');
    v = y_t - Z_t * a_t - b_t;

    K = F * P_t * Z_t' / S;
    L = F - K * Z_t;

    r_t = Z_t' / S * v + L' * r_t;
    r(:, t) = r_t;

    eta_hat(:, t) = Q_noise * R_shock' * r_t;
end

% ========================
% State smoother
% ========================
alpha_hat = zeros(Nz, T);
ah_t = z0 + P0 * r(:, 1);
alpha_hat(:, 1) = ah_t;

for t = 2:T
    ah_t = F * ah_t + R_shock * Q_noise * R_shock' * r(:, t);
    alpha_hat(:, t) = ah_t;
end

% --- Pack output ---
results.SmoothedStates = alpha_hat;
results.SmoothedShocks = eta_hat;
results.Pred           = pred;
results.VPred          = vpred;

end
