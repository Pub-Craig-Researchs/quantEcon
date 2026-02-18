classdef Simplex
    % SIMPLEX Simplified Simplex Solver for SDID weights
    %
    %   Minimizes || X*w - target || + zeta * ||w||
    %   Subject to sum(w)=1, w >= 0.

    methods (Static)
        function w = solve(X, target, zeta)
            % SOLVE Solve constrained least squares
            % Inputs:
            %   X: (T x N) matrix (Control units)
            %   target: (T x 1) vector (Treated unit average)
            %   zeta: regularization parameter

            [T, N] = size(X);

            % Formulation as Quadratic Programming
            % min 0.5 * w' H w + f' w
            % Obj = || Xw - y ||^2 + zeta ||w||^2
            %     = (Xw - y)'(Xw - y) + zeta w'w
            %     = w'X'Xw - 2y'Xw + y'y + zeta w'w
            %     = w' (X'X + zeta*I) w - 2 (y'X) w

            H = (X' * X) + zeta * eye(N);
            f = - (target' * X)'; % Column vector

            % Constraints
            % 1. sum(w) = 1  => Aeq * w = beq
            Aeq = ones(1, N);
            beq = 1;

            % 2. w >= 0      => lb
            lb = zeros(N, 1);
            ub = inf(N, 1);

            % Solve
            options = optimoptions('quadprog', 'Display', 'off');
            w = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);
        end
    end
end
