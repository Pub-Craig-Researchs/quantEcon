classdef CausalTest < matlab.unittest.TestCase

    methods (Test)
        function testSharpRDD(testCase)
            % Simulate Sharp RDD
            % Cutoff = 0
            % Y = 1 + 2*D + X + e
            N = 1000;
            rng(123);
            X = randn(N, 1);
            D = double(X >= 0);
            Y = 1 + 2*D + 0.5*X + 0.1*randn(N, 1);

            rdd = quantecon.causal.Rdd();
            res = rdd.estimate(X, Y);

            testCase.verifyEqual(res.Coefficients, 2.0, 'AbsTol', 0.1, 'Sharp RDD effect incorrect');
        end

        function testFuzzyRDD(testCase)
            % Simulate Fuzzy RDD
            % Cutoff = 0
            % P(D=1) jumps from 0.1 to 0.9 at X=0
            N = 2000;
            rng(456);
            X = randn(N, 1);
            prob = 0.1 + 0.8 * (X >= 0);
            D = binornd(1, prob);

            % LATE = 2
            Y = 1 + 2*D + 0.5*X + 0.1*randn(N, 1);

            rdd = quantecon.causal.Rdd();
            res = rdd.estimate(X, Y, D);

            testCase.verifyEqual(res.FirstStage, 0.8, 'AbsTol', 0.1, 'First Stage incorrect');
            testCase.verifyEqual(res.Coefficients, 2.0, 'AbsTol', 0.2, 'Fuzzy RDD effect incorrect');
        end

        function testBandwidthSelection(testCase)
            % Test that bandwidth selectors run without error
            N = 500;
            X = randn(N, 1);
            Y = (X >= 0) + X + 0.1*randn(N, 1);

            rdd = quantecon.causal.Rdd();
            rdd.BandwidthMethod = "cct";
            res = rdd.estimate(X, Y);
            testCase.verifyNotEmpty(res.Bandwidth);

            rdd.BandwidthMethod = "rot";
            res = rdd.estimate(X, Y);
            testCase.verifyNotEmpty(res.Bandwidth);
        end
    end
end
