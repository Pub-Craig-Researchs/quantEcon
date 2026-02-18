function obj=no_legend(obj)
% no_legend() remove side legend on the plot
%
% Useful when plotting multiple Gramm objects with the same
% legend

obj.layout_options.legend=false;

end
