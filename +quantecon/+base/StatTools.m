classdef StatTools
    % STATTOOLS Statistical utility functions
    %
    %   Static Methods:
    %       moments - Calculate first four moments

    methods (Static)
        function stats = moments(x)
            % MOMENTS Calculate mean, variance, skewness, and kurtosis
            %
            %   Usage:
            %       stats = quantecon.base.StatTools.moments(x)

            arguments
                x (:,:) double
            end

            stats.mean = mean(x, 'omitnan');
            stats.var = var(x, 'omitnan');
            stats.skewness = skewness(x, 0, 'omitnan'); % 0 = bias corrected
            stats.kurtosis = kurtosis(x, 0, 'omitnan');
        end
    end
end
