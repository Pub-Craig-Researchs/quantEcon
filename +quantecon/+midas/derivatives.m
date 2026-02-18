function [f, grad, H, weights] = derivatives(params, y, X_hf, V_lf, poly, beta_case)
%DERIVATIVES Analytical Gradient and Hessian for MIDAS functions
% [f, grad, H, weights] = derivatives(...)

p = size(X_hf, 2);
k_lf = size(V_lf, 2);

switch poly
    case "Beta"
        theta1 = params(1);
        theta2 = params(2);
        theta3 = params(3);
        beta = params(4);
        rho = params(5:end);

        x_grid = (1:p)' / (p + 1);
        ii = ones(p, 1);

        % m = gamma(theta1+theta2)/(gamma(theta1)*gamma(theta2))
        m = exp(gammaln(theta1 + theta2) - gammaln(theta1) - gammaln(theta2));
        weights = (x_grid.^(theta1-1) .* (ii - x_grid).^(theta2-1)) * m + ii * theta3;

    case "ExpAlmon"
        theta1 = params(1);
        theta2 = params(2);
        beta = params(3);
        rho = params(4:end);

        xi = (1:p)';
        xi_sq = xi.^2;
        weights = exp(theta1 .* xi + theta2 .* xi_sq);
end

% Residue and Objective
resid = y - X_hf * (beta * weights) - V_lf * rho;
f = resid' * resid;

if nargout > 1
    % GRADIENT
    nabla_G = 2 * resid;

    if poly == "Beta"
        % Ported and optimized from resource
        T_star_w_C = -beta * X_hf';
        T_star_beta_C = -weights' * X_hf';
        T_star_rho_C = -V_lf';

        weights_tilde = (x_grid.^(theta1-1) .* (ii - x_grid).^(theta2-1)) * m;

        % theta2 derivative
        log_der_t2 = log(ii - x_grid) + psi(theta1 + theta2) - psi(theta2);
        T_star_t2_W = (log_der_t2 .* weights_tilde)';
        part2 = T_star_t2_W * T_star_w_C * nabla_G;

        % theta1 & theta3 derivatives based on beta_case
        part1 = 0; part3 = 0;
        if beta_case == 0 || beta_case == 2
            log_der_t1 = log(x_grid) + psi(theta1 + theta2) - psi(theta1);
            T_star_t1_W = (log_der_t1 .* weights_tilde)';
            part1 = T_star_t1_W * T_star_w_C * nabla_G;
        end
        if beta_case == 0 || beta_case == 1
            T_star_t3_W = ii';
            part3 = T_star_t3_W * T_star_w_C * nabla_G;
        end

        part4 = T_star_beta_C * nabla_G;
        part5 = T_star_rho_C * nabla_G;
        grad = [part1; part2; part3; part4; part5];

    else % ExpAlmon
        T_star_w_C = -beta * X_hf';
        T_star_beta_C = -weights' * X_hf';
        T_star_rho_C = -V_lf';

        T_star_t1_W = (weights .* xi)';
        T_star_t2_W = (weights .* xi_sq)';

        part1 = T_star_t1_W * T_star_w_C * nabla_G;
        part2 = T_star_t2_W * T_star_w_C * nabla_G;
        part3 = T_star_beta_C * nabla_G;
        part4 = T_star_rho_C * nabla_G;
        grad = [part1; part2; part3; part4];
    end
end

if nargout > 2
    % HESSIAN
    if poly == "Beta"
        % E1 = 2*(y-Z * beta * weights - V*rho )
        E1 = nabla_G;
        F1 = -beta * X_hf';
        D1 = log(x_grid) + psi(theta1 + theta2) - psi(theta1);
        C1 = weights_tilde;

        dC1_t1 = C1 .* D1;
        dD1_t1 = psi(1, theta1 + theta2) - psi(1, theta1);
        dB1_t1 = F1 * (-2 * (X_hf * (beta * dC1_t1)));

        D2 = log(ii - x_grid) + psi(theta1 + theta2) - psi(theta2);
        dC1_t2 = C1 .* D2;
        dD1_t2 = psi(1, theta1 + theta2);
        dB1_t2 = F1 * (-2 * (X_hf * (beta * dC1_t2)));

        dE1_t3 = -2 * (X_hf * (beta * ii));
        dF1_beta = -X_hf';
        dE1_beta = -2 * (X_hf * weights);
        dE1_rho = -2 * V_lf;

        % line 2
        dC2_t2 = C1 .* D2;
        dD2_t2 = psi(1, theta1 + theta2) - psi(1, theta2);

        H22 = (dC2_t2 .* D2 + C1 .* dD2_t2)' * (F1 * E1) + (C1 .* D2)' * dB1_t2;
        H23 = (C1 .* D2)' * (F1 * dE1_t3);
        H24 = (C1 .* D2)' * (dF1_beta * E1 + F1 * dE1_beta);
        H25 = (C1 .* D2)' * (F1 * dE1_rho);

        A3 = ii' * F1;
        dA3_beta = ii' * dF1_beta;

        % Partial Hessian construction based on beta_case
        H11 = 0; H12 = 0; H13 = 0; H14 = 0; H15 = zeros(1, k_lf);
        H33 = 0; H34 = 0; H35 = zeros(1, k_lf);

        if beta_case == 0 || beta_case == 2
            H11 = (dC1_t1 .* D1 + C1 .* dD1_t1)' * (F1 * E1) + (C1 .* D1)' * dB1_t1;
            H12 = (dC1_t2 .* D1 + C1 .* dD1_t2)' * (F1 * E1) + (C1 .* D1)' * dB1_t2;
            H14 = (C1 .* D1)' * (dF1_beta * E1 + F1 * dE1_beta);
            H15 = (C1 .* D1)' * (F1 * dE1_rho);
        end

        if beta_case == 0 || beta_case == 1
            H13 = (C1 .* D1)' * (F1 * dE1_t3);
            H33 = A3 * dE1_t3;
            H34 = dA3_beta * E1 + A3 * dE1_beta;
            H35 = A3 * dE1_rho;
        end

        H44 = (-weights' * X_hf') * dE1_beta;
        H45 = (-weights' * X_hf') * dE1_rho;
        H55 = -V_lf' * dE1_rho;

        H = [H11, H12, H13, H14, H15;
            H12, H22, H23, H24, H25;
            H13, H23, H33, H34, H35;
            H14, H24, H34, H44, H45;
            H15', H25', H35', H45', H55];

    else % ExpAlmon
        A1_2 = -beta * X_hf';
        B1 = 2 * resid;
        dA1_t1 = (weights .* xi_sq)' * A1_2;
        dB1_t1 = -2 * beta * (X_hf * (weights .* xi));
        dA1_t2 = (weights .* xi.^3)' * A1_2;
        dB1_t2 = -2 * beta * (X_hf * (weights .* xi_sq));
        dA1_beta = (weights .* xi)' * (-X_hf');
        dB1_beta = -2 * (X_hf * weights);
        dB1_rho = -2 * V_lf;

        A1 = (weights .* xi)' * A1_2;
        H11 = dA1_t1 * B1 + A1 * dB1_t1;
        H12 = dA1_t2 * B1 + A1 * dB1_t2;
        H13 = dA1_beta * B1 + A1 * dB1_beta;
        H14 = A1 * dB1_rho;

        A2 = (weights .* xi_sq)' * A1_2;
        dA2_t2 = (weights .* xi.^4)' * A1_2;
        dA2_beta = (weights .* xi_sq)' * (-X_hf');
        H22 = dA2_t2 * B1 + A2 * dB1_t2;
        H23 = dA2_beta * B1 + A2 * dB1_beta;
        H24 = A2 * dB1_rho;

        A3 = -weights' * X_hf';
        H33 = A3 * dB1_beta;
        H34 = A3 * dB1_rho;

        H44 = -V_lf' * dB1_rho;

        H = [H11, H12, H13, H14;
            H12, H22, H23, H24;
            H13, H23, H33, H34;
            H14', H24', H34', H44];
    end
end

end
