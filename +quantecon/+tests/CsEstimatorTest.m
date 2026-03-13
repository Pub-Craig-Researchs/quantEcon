classdef CsEstimatorTest < matlab.unittest.TestCase

    methods (Test)
        function testSyntheticStaggered(testCase)
            % Generate synthetic staggered adoption data
            N = 100;
            T = 10;

            % IDs and Time
            ids = repmat((1:N)', T, 1);
            time = kron((1:T)', ones(N, 1));

            % Staggered Groups
            % Group 1: Treated at T=5 (Units 1-30)
            % Group 2: Treated at T=7 (Units 31-60)
            % Control: Never treated (Units 61-100)

            g = inf(N, 1);
            g(1:30) = 5;
            g(31:60) = 7;

            % Treatment Indicator
            % Expand g to match ids
            G_long = g(ids);
            treat = double(time >= G_long);

            % Outcome y
            % y = i + t + tau * treat + e
            % tau = 1 (constant treatment effect)
            rng(42);
            y = ids + time + 1.0 * treat + 0.5 * randn(N*T, 1);

            % Estimate
            mdl = quantecon.panel.did.CsEstimator();
            mdl = mdl.estimate(y, treat, time, ids);

            % Check Results
            res = mdl.aggregate('Simple');

            % Check that estimate is close to 1.0
            testCase.verifyEqual(res.Estimate, 1.0, 'AbsTol', 0.2);

            % Check Event Study
            es = mdl.aggregate('EventStudy');
            % Check that pre-event coefficients (e < 0) are close to 0
            pre_mask = (es.EventTime < 0);
            testCase.verifyTrue(all(abs(es.Estimate(pre_mask)) < 0.3));
        end

        function testOutcomeRegression(testCase)
            % Test Outcome Regression with covariates
            N = 100;
            T = 5;

            ids = repmat((1:N)', T, 1);
            time = kron((1:T)', ones(N, 1));

            % Covariates X (Time Invariant, N x 2)
            rng(1);
            X_N = randn(N, 2);
            X = repmat(X_N, T, 1); % Expand for input check but model takes N x K
            % Actually model supports N x K or N*T x K?
            % My implementation checks size. If N x K it works.

            g = inf(N, 1);
            g(1:50) = 3; % Treated at T=3

            G_long = g(ids);
            treat = double(time >= G_long);

            % Outcome depends on X
            % y = treat + X*beta + e
            y = treat * 1.0 + X_N(ids, 1) * 2.0 + randn(N*T, 1) * 0.1;

            mdl = quantecon.panel.did.CsEstimator();
            mdl.Method = "OR";
            % Pass X as N x K
            mdl = mdl.estimate(y, treat, time, ids, X_N);

            res = mdl.aggregate('Simple');
            testCase.verifyEqual(res.Estimate, 1.0, 'AbsTol', 0.2);
        end
    end
end
