function results = CopulaFit(u, family)
%COPULAFIT Fit Copula models to uniform margins
%
%   Usage:
%       results = quantecon.multivariate.CopulaFit(u, 'Gaussian')
%       results = quantecon.multivariate.CopulaFit(u, 't')
%       results = quantecon.multivariate.CopulaFit(u, 'Clayton') % Bivariate only
%
%   Inputs:
%       u      - (T x K) matrix of uniform margins in (0,1)
%       family - (string) 'Gaussian', 't', 'Clayton', 'SJC'
%
%   Outputs:
%       results - Struct with estimated parameters and log-likelihood.

arguments
    u (:,:) double {mustBeNumeric, mustBeReal}
    family (1,1) string {mustBeMember(family, ["Gaussian", "t", "Clayton", "SJC"])}
end

results = struct();
results.Family = family;
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

switch family
    case "Gaussian"
        z_norm = norminv(u);
        Rho = corr(z_norm);
        results.Parameters = Rho;
        results.LogLikelihood = gaussian_copula_ll(Rho, z_norm);

    case "t"
        obj = @(nu) -t_copula_ll_nu(nu, u);
        nu_est = fmincon(obj, 5, [], [], [], [], 2.01, 100, [], options);
        results.Parameters.Nu = nu_est;
        results.Parameters.Rho = corr(tinv(u, nu_est));
        results.LogLikelihood = t_copula_ll_nu(nu_est, u);

    case "Clayton"
        if size(u, 2) ~= 2, error("Clayton copula in this implementation is bivariate only."); end
        tau = corr(u(:,1), u(:,2), 'type', 'Kendall');
        theta_init = max(0.1, 2 * tau / (1 - tau));
        obj = @(theta) -clayton_ll(theta, u);
        theta_est = fmincon(obj, theta_init, [], [], [], [], 0.001, 100, [], options);
        results.Parameters = theta_est;
        results.LogLikelihood = clayton_ll(theta_est, u);

    case "SJC"
        if size(u, 2) ~= 2, error("SJC copula in this implementation is bivariate only."); end
        x0 = [0.1; 0.1];
        lb = [1e-6; 1e-6];
        ub = [0.999; 0.999];
        obj = @(p) -sjc_ll(p(1), p(2), u);
        p_est = fmincon(obj, x0, [], [], [], [], lb, ub, [], options);
        results.Parameters.TauU = p_est(1);
        results.Parameters.TauL = p_est(2);
        results.LogLikelihood = sjc_ll(p_est(1), p_est(2), u);
end
end

function ll = gaussian_copula_ll(Rho, z)
[T, K] = size(z);
try
    det_R = det(Rho);
    ll_val = 0;
    for t = 1:T
        % Use x / Rho * x' instead of x * inv(Rho) * x'
        ll_val = ll_val - 0.5 * log(det_R) - 0.5 * (z(t,:) / Rho * z(t,:)') + 0.5 * (z(t,:) * z(t,:)');
    end
    ll = ll_val;
catch
    ll = -1e10;
end
end

function ll = t_copula_ll_nu(nu, u)
[T, K] = size(u);
z = tinv(u, nu);
Rho = corr(z);
try
    det_R = det(Rho);
    term1 = T * (gammaln((nu + K) / 2) + (K - 1) * gammaln(nu / 2) - K * gammaln((nu + 1) / 2));
    term2 = -0.5 * T * log(det_R);
    term3 = 0;
    for t = 1:T
        term3 = term3 - ((nu + K) / 2) * log(1 + (z(t,:) / Rho * z(t,:)') / nu);
        term3 = term3 + ((nu + 1) / 2) * sum(log(1 + z(t,:).^2 / nu));
    end
    ll = term1 + term2 + term3;
catch
    ll = -1e10;
end
end

function ll = clayton_ll(theta, u)
T = size(u, 1);
u1 = u(:,1); u2 = u(:,2);
term1 = T * log(1 + theta);
term2 = -(1 + theta) * sum(log(u1 .* u2));
term3 = -(2 + 1/theta) * sum(log(u1.^-theta + u2.^-theta - 1));
ll = term1 + term2 + term3;
end

function ll = sjc_ll(tauU, tauL, u)
u1 = u(:,1); u2 = u(:,2);
k1 = 1 ./ log2(2 - tauU);
k2 = -1 ./ log2(tauL);
pdf1 = sjc_pdf_part(u1, u2, k1, k2);
k1_L = 1 ./ log2(2 - tauL);
k2_U = -1 ./ log2(tauU);
pdf2 = sjc_pdf_part(1 - u1, 1 - u2, k1_L, k2_U);
ll = sum(log(0.5 * (pdf1 + pdf2)));
end

function out = sjc_pdf_part(u, v, k1, k2)
JC1 = (k1.*k2.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 1).*(1./k2 + 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2 + 2));
JC2 = (k1.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 2).*(1./k1 - 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(2./k2 + 2));
out = JC1 - JC2;
out(out <= 0) = 1e-10;
end
