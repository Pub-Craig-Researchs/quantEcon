classdef Mes < handle
    %MES Marginal Expected Shortfall
    %
    %   Implements MES using DCC-GARCH framework (Brownlees & Engle 2012).
    %
    %   MES = E[R_inst | R_sys < C]
    %       = sigma_inst * rho * E[e_sys | e_sys < C/sigma_sys] + ...
    %
    %   Usage:
    %       mdl = quantecon.risk.Mes();
    %       res = mdl.estimate(R_sys, R_inst);

    properties
        Quantile (1,1) double = 0.05
        Results struct
    end

    methods
        function obj = Mes(q)
            if nargin > 0
                obj.Quantile = q;
            end
        end

        function results = estimate(obj, R_sys, R_inst)
            %ESTIMATE Estimate MES
            %
            %   Input:
            %       R_sys: (Tx1) System returns
            %       R_inst: (TxK) Institution returns (Matrix allowed)

            arguments
                obj
                R_sys (:,1) double
                R_inst (:,:) double
            end

            [T, K] = size(R_inst);

            % 1. Estimate DCC-GARCH between R_sys and each R_inst_k
            % We need to loop because DCC is typically multivariate.
            % Or we can put all in one big DCC if K is small.
            % For efficiency + simplicity: Bivariate DCC for each pair (Sys, Inst_k)

            mes_est = zeros(T, K);
            lrmes_est = zeros(T, K);
            beta_est = zeros(T, K);

            for k = 1:K
                Y = [R_sys, R_inst(:,k)];

                % Use our new Dcc class
                dcc = quantecon.multivariate.Dcc('M', 1, 'N', 1); % Bivariate DCC(1,1)
                res = dcc.estimate(Y);

                % Extract Volatilities and Correlations
                % Ht is (2 x 2 x T)
                % Ht(1,1,t) = sigma_sys^2
                % Ht(2,2,t) = sigma_inst^2
                % Rt(1,2,t) = rho

                Ht = res.Ht;
                Rt = res.Rt;

                sigma_sys = sqrt(squeeze(Ht(1,1,:)));
                sigma_inst = sqrt(squeeze(Ht(2,2,:)));
                rho = squeeze(Rt(1,2,:));

                % Beta_inst = rho * sigma_inst / sigma_sys
                beta_t = rho .* (sigma_inst ./ sigma_sys);
                beta_est(:,k) = beta_t;

                % 2. Calculate MES
                % Scaillet (2004) / Brownlees Engle (2012)
                % MES_t = sigma_inst_t * rho_t * E[eps_m | eps_m < c] + sigma_inst_t * sqrt(1-rho_t^2) * E[eps_i | eps_m < c]
                % Assuming standard normal innovations:
                % E[eps_m | eps_m < c] = -pdf(c) / cdf(c)
                % Inner expectation of eps_i is 0 if bivariate normal

                % C = VaR of system innovations
                % If we define the tail event as R_sys < VaR_sys(q)
                % Then eps_m < VaR_sys(q) / sigma_sys = quantile(std_resid, q)

                % Using simple approx for standard normal
                alpha = obj.Quantile;
                c = norminv(alpha);

                tail_exp_sys = -normpdf(c) / alpha;

                % MES_t = sigma_inst_t * rho_t * tail_exp_sys
                % (This ignores the second term which is 0 for normal)

                mes_est(:,k) = sigma_inst .* rho .* tail_exp_sys;

                % LRMES (Long Run MES) - Acharya et al (2012) approx
                % LRMES = 1 - exp(18 * MES) ... or similar.
                % Using the approximation formula from SRISK papers:
                % LRMES ~= 1 - exp( log(1-d) * beta )
                % where d is crisis threshold (e.g. 40% drop)
                d = 0.40;
                lrmes_est(:,k) = 1 - exp(log(1-d) * beta_t);
            end

            results.MES = mes_est;
            results.LRMES = lrmes_est;
            results.Beta = beta_est;

            obj.Results = results;
        end
    end
end
