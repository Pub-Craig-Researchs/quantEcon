classdef ConnectednessVis
    %CONNECTEDNESSVIS Visualization Tools for Connectedness
    %   Provides dynamic plots and network graphs for quantecon.finance.Connectedness
    %
    %   Reference: Gabauer, D. (2022). "ConnectednessApproach package"
    %
    %   Methods:
    %       plotTCI(cnObj)      - Plots the Total Connectedness Index
    %       plotNET(cnObj)      - Plots dynamic Net Directional Connectedness
    %       plotDirectional(...) - Plots 'TO' or 'FROM' dynamic connectedness
    %       plotNetwork(cnObj)   - Plots the Net Pairwise Directional Connectedness graph

    methods(Static)
        function plotTCI(cnObj)
            %PLOTTCI Visualize dynamic Total Connectedness Index
            tci = cnObj.Indices.Total;
            t_len = size(tci, 1);
            if t_len == 1
                fprintf('Static Connectedness: TCI = %.2f%%\n', tci);
                return;
            end

            t = cnObj.Time;
            if length(t) > t_len
                t = t((end - t_len + 1):end);
            end

            if isnumeric(t) && isempty(t)
                t = (1:length(tci))';
            end

            figure('Name', 'Total Connectedness Index', 'Color', 'w', 'Position', [100 100 800 400]);

            g = quantecon.vis.Gramm('x', t, 'y', tci);
            g.geom_line();
            g.set_names('x', '', 'y', 'TCI (%)');
            g.set_title('Total Connectedness Index (TCI)');
            g.set_color_options('map', [0.1 0.1 0.1]);
            g.axe_property('TickDir', 'out', 'Box', 'off', 'XGrid', 'on', 'YGrid', 'on');
            g.draw();
        end

        function plotNET(cnObj)
            %PLOTNET Visualize dynamic Net Connectedness
            net = cnObj.Indices.Net;
            t_len = size(net, 1);
            if t_len == 1
                disp('Static Connectedness: No dynamic NET to plot.');
                return;
            end

            t = cnObj.Time;
            if length(t) > t_len
                t = t((end - t_len + 1):end);
            end

            N = size(net, 2);

            if istable(cnObj.Data)
                names = cnObj.Data.Properties.VariableNames;
            else
                names = arrayfun(@(x) sprintf('Var%d', x), 1:N, 'UniformOutput', false);
            end

            TimeData = repmat(t, N, 1);
            ValData = net(:);
            % Create grouped labels matching the T x N shape
            Groups = repmat(names(:)', length(t), 1);
            GroupsData = Groups(:);

            figure('Name', 'Net Directional Connectedness', 'Color', 'w', 'Position', [100 100 1000 600]);

            g = quantecon.vis.Gramm('x', TimeData, 'y', ValData);
            g.facet_wrap(GroupsData, 'ncols', ceil(sqrt(N)));
            g.geom_line();
            g.geom_hline('yintercept', 0, 'style', 'k--');
            g.set_names('x', '', 'y', '', 'column', '');
            g.set_title('Net Directional Connectedness');
            g.set_color_options('map', [0.2 0.5 0.7]); % Aesthetic R-like blue
            g.axe_property('TickDir', 'out', 'XGrid', 'on', 'YGrid', 'on');
            g.draw();
        end

        function plotDirectional(cnObj, type)
            %PLOTDIRECTIONAL Visualize dynamic 'TO' or 'FROM' Connectedness
            arguments
                cnObj
                type (1,1) string {mustBeMember(type, ["TO", "FROM", "to", "from"])}
            end

            if strcmpi(type, "TO")
                data = cnObj.Indices.To;
                title_prefix = 'TO ';
            else
                data = cnObj.Indices.From;
                title_prefix = 'FROM ';
            end

            t_len = size(data, 1);
            if t_len == 1
                disp(['Static Connectedness: No dynamic ', title_prefix, 'to plot.']);
                return;
            end

            t = cnObj.Time;
            if length(t) > t_len
                t = t((end - t_len + 1):end);
            end

            N = size(data, 2);
            if istable(cnObj.Data)
                names = cnObj.Data.Properties.VariableNames;
            else
                names = arrayfun(@(x) sprintf('Var%d', x), 1:N, 'UniformOutput', false);
            end

            TimeData = repmat(t, N, 1);
            ValData = data(:);
            % Create prefixed grouped labels
            group_names = cellfun(@(x) [title_prefix, x], names, 'UniformOutput', false);
            Groups = repmat(group_names(:)', length(t), 1);
            GroupsData = Groups(:);

            figure('Name', sprintf('%s Directional Connectedness', upper(type)), 'Color', 'w', 'Position', [100 100 1000 600]);

            g = quantecon.vis.Gramm('x', TimeData, 'y', ValData);
            g.facet_wrap(GroupsData, 'ncols', ceil(sqrt(N)));
            g.geom_line();
            g.set_names('x', '', 'y', '', 'column', '');
            g.set_title(sprintf('%s Directional Connectedness', upper(type)));
            g.set_color_options('map', [0.2 0.5 0.7]); % Aesthetic R-like blue
            g.axe_property('TickDir', 'out', 'XGrid', 'on', 'YGrid', 'on');
            g.draw();
        end

        function plotNetwork(cnObj, options)
            %PLOTNETWORK Visualize Net Pairwise Directional Connectedness
            arguments
                cnObj
                options.Threshold (1,1) double = 0.0
                options.Method (1,1) string {mustBeMember(options.Method, ["NPDC", "PCI", "npdc", "pci"])} = "NPDC"
            end

            x = cnObj.Indices.Pairwise; % T x N x N
            if size(x, 1) > 1
                x_mean = squeeze(mean(x, 1));
            else
                x_mean = squeeze(x);
            end

            N = size(x_mean, 1);
            if istable(cnObj.Data)
                names = cnObj.Data.Properties.VariableNames;
            else
                names = arrayfun(@(x) sprintf('Var%d', x), 1:N, 'UniformOutput', false);
            end

            if strcmpi(options.Method, "NPDC")
                val = x_mean - x_mean';
                val(val < options.Threshold) = 0;
                title_str = 'Net Pairwise Directional Connectedness';
            else
                % Pairwise Connectedness Index (symmetrical)
                % PCI_ij = 2 * (C_ij + C_ji) / (C_ii + C_ij + C_ji + C_jj)
                val = zeros(N, N);
                for i = 1:N
                    for j = 1:N
                        if i ~= j
                            denom = x_mean(i,i) + x_mean(i,j) + x_mean(j,i) + x_mean(j,j);
                            val(i,j) = 200 * (x_mean(i,j) + x_mean(j,i)) / denom;
                        end
                    end
                end
                val(val < options.Threshold) = 0;
                val = triu(val, 1); % undirected
                title_str = 'Pairwise Connectedness Index';
            end

            if strcmpi(options.Method, "NPDC")
                G = digraph(val, names);
                is_directed = true;
            else
                G = graph(val, names, 'omitselfloops');
                is_directed = false;
            end

            % Edge widths
            max_weight = max(G.Edges.Weight);
            if max_weight == 0; max_weight = 1; end
            LWidths = 5 * G.Edges.Weight / max_weight;

            % Node Sizes
            if is_directed
                out_degree = sum(val, 2);
            else
                out_degree = sum(val + val', 2);
            end

            max_deg = max(out_degree);
            if max_deg == 0; max_deg = 1; end
            MarkerSizes = 8 + 20 * (out_degree / max_deg);

            figure('Name', title_str, 'Color', 'w', 'Position', [100 100 800 800]);

            p = plot(G, 'Layout', 'circle');

            % Style Nodes
            p.NodeColor = [0.2 0.5 0.7]; % steelblue equivalent
            p.MarkerSize = MarkerSizes;
            p.NodeFontSize = 11;
            p.NodeFontWeight = 'bold';

            % Highlight nodes with high outgoing connections
            if max_deg > 1
                highlight_idx = out_degree > (0.5 * max_deg);
                highlight(p, highlight_idx, 'NodeColor', [0.8 0.6 0.2]); % gold equivalent
            end

            % Style Edges
            p.EdgeColor = [0.6 0.6 0.6];
            p.LineWidth = max(0.2, LWidths);

            if is_directed
                p.EdgeAlpha = 0.5;
                p.ArrowSize = 15;
            end

            title(title_str, 'FontSize', 16, 'FontWeight', 'bold');
            axis off;
        end
    end
end
