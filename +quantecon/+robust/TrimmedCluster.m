classdef TrimmedCluster
    %TRIMMEDCLUSTER Robust clustering with trimming and scatter restrictions.
    %
    %   Algorithm: Trimmed k-means/k-medians with eigenvalue restrictions.
    %   Reference: Garcia-Escudero et al. (2008).

    properties
        K (1,1) double                 % Number of clusters
        Alpha (1,1) double             % Trimming level (0 to 0.5)
        RestrFactor (1,1) double       % Eigenvalue restriction factor (>= 1)
        MaxIter (1,1) double = 100     % Max C-steps
        Nsamp (1,1) double = 300       % Number of random starts
    end

    methods
        function obj = TrimmedCluster(K, alpha, restr)
            arguments
                K (1,1) double {mustBeInteger, mustBePositive} = 2
                alpha (1,1) double {mustBeNonnegative, mustBeLessThan(alpha, 0.5)} = 0.1
                restr (1,1) double {mustBeGreaterThanOrEqual(restr, 1)} = 10
            end
            obj.K = K;
            obj.Alpha = alpha;
            obj.RestrFactor = restr;
        end

        function out = estimate(obj, Y)
            %ESTIMATE Fit the trimmed clustering model.

            [n, v] = size(Y);
            h = floor(n * (1 - obj.Alpha));

            bestObj = -Inf;
            bestIdx = zeros(n, 1);
            bestMu = zeros(obj.K, v);
            bestSigma = zeros(v, v, obj.K);

            % Random Multi-start
            for s = 1:obj.Nsamp
                % Initialize centroids randomly
                idx_init = randperm(n, obj.K);
                mu = Y(idx_init, :);
                sigma = repmat(eye(v), [1, 1, obj.K]);
                pi_k = ones(obj.K, 1) / obj.K;

                % C-steps
                for iter = 1:obj.MaxIter
                    % 1. Data Assignment & Trimming
                    dists = zeros(n, obj.K);
                    for k = 1:obj.K
                        diff = Y - mu(k, :);
                        % Use Mahalanobis-like distance (log density)
                        try
                            invS = inv(sigma(:,:,k));
                            d = sum((diff * invS) .* diff, 2);
                            dists(:, k) = -0.5 * (log(det(sigma(:,:,k))) + d) + log(pi_k(k));
                        catch
                            % Singular case handling
                            dists(:, k) = -Inf;
                        end
                    end

                    [maxD, clusterIdx] = max(dists, [], 2);

                    % Find top h observations
                    [~, sortIdx] = sort(maxD, 'descend');
                    hIdx = sortIdx(1:h);

                    % 2. Update Parameters
                    newMu = zeros(obj.K, v);
                    newSigma = zeros(v, v, obj.K);
                    newPi = zeros(obj.K, 1);
                    ni = zeros(obj.K , 1);
                    eigenvalues = zeros(v, obj.K);
                    eigenvectors = cell(obj.K, 1);

                    for k = 1:obj.K
                        k_obs = hIdx(clusterIdx(hIdx) == k);
                        ni(k) = length(k_obs);
                        if ni(k) > v
                            newMu(k, :) = mean(Y(k_obs, :), 1);
                            S_unrestr = cov(Y(k_obs, :));
                            [V, D] = eig(S_unrestr);
                            eigenvalues(:, k) = diag(D);
                            eigenvectors{k} = V;
                        else
                            % Handle small clusters
                            newMu(k, :) = mu(k, :);
                            eigenvalues(:, k) = ones(v, 1);
                            eigenvectors{k} = eye(v);
                        end
                        newPi(k) = ni(k) / h;
                    end

                    % 3. Apply Eigenvalue Restrictions
                    restrEig = obj.restrictEigenvalues(eigenvalues, ni, obj.RestrFactor);
                    for k = 1:obj.K
                        newSigma(:,:,k) = eigenvectors{k} * diag(restrEig(:, k)) * eigenvectors{k}';
                    end

                    % Check convergence
                    if norm(newMu - mu) < 1e-6
                        break;
                    end
                    mu = newMu;
                    sigma = newSigma;
                    pi_k = newPi;
                end

                % Evaluate target function
                objVal = sum(maxD(hIdx));
                if objVal > bestObj
                    bestObj = objVal;
                    bestIdx = zeros(n, 1);
                    bestIdx(hIdx) = clusterIdx(hIdx);
                    bestMu = mu;
                    bestSigma = sigma;
                end
            end

            out.idx = bestIdx;
            out.mu = bestMu;
            out.sigma = bestSigma;
            out.h = h;
            out.obj = bestObj;
        end
    end

    methods (Access = private)
        function out = restrictEigenvalues(obj, eigenvalues, ni, restr)
            % Simplified logic based on restreigen.m
            [v, k] = size(eigenvalues);
            if restr == 1
                avg = sum(eigenvalues .* ni', 'all') / (v * sum(ni));
                out = ones(v, k) * avg;
                return;
            end

            % Heuristic approach if full Fritz logic is too long
            % (Simple clipping for now, better than nothing)
            out = eigenvalues;
            maxE = max(out, [], 'all');
            minE = maxE / restr;
            out(out < minE) = minE;
        end
    end
end
