classdef Srisk < handle
    %SRISK Systemic Risk (Capital Shortfall)
    %
    %   SRISK = k * Liabilities - (1-k) * (1 - LRMES) * Equity
    %
    %   Represents the capital shortfall of a firm conditional on a systemic crisis.
    %   Acharya, Engle, Richardson (2012).
    %
    %   Usage:
    %       mdl = quantecon.risk.Srisk();
    %       res = mdl.estimate(R_sys, R_inst, Liabilities, Equity);

    properties
        PrudentialRatio (1,1) double = 0.08 % k (e.g., 8%)
        Quantile (1,1) double = 0.05
        Results struct
    end

    methods
        function obj = Srisk(k)
            if nargin > 0
                obj.PrudentialRatio = k;
            end
        end

        function results = estimate(obj, R_sys, R_inst, Liabilities, Equity)
            %ESTIMATE Estimate SRISK
            %
            %   Input:
            %       R_sys: (Tx1) System returns
            %       R_inst: (TxK) Institution returns
            %       Liabilities: (TxK) Book value of Debt
            %       Equity: (TxK) Market value of Equity (Market Cap)

            arguments
                obj
                R_sys (:,1) double
                R_inst (:,:) double
                Liabilities (:,:) double
                Equity (:,:) double
            end

            [T, K] = size(R_inst);
            assert(all(size(Liabilities) == [T, K]), 'Liabilities size mismatch');
            assert(all(size(Equity) == [T, K]), 'Equity size mismatch');

            % 1. Calculate LRMES
            mes_mdl = quantecon.risk.Mes(obj.Quantile);
            res_mes = mes_mdl.estimate(R_sys, R_inst);

            LRMES = res_mes.LRMES;

            % 2. Calculate SRISK
            % SRISK = k * Debt - (1-k) * (1 - LRMES) * Equity

            k_ratio = obj.PrudentialRatio;

            % Note: Liabilities here should be Book Liabilities.
            % If inputs are Debt, use Debt.

            SRISK = k_ratio * Liabilities - (1 - k_ratio) * (1 - LRMES) .* Equity;

            % SRISK is usually floored at 0 (Surplus doesn't offset others in aggregate system view,
            % but for individual firm positive surplus is negative risk)
            % However, standard reporting usually shows positive values as Risk.
            % We return raw values.

            results.SRISK = SRISK;
            results.LRMES = LRMES;
            results.MES_Results = res_mes; % Store underlying MES results

            obj.Results = results;
        end
    end
end
