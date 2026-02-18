function obj = fig(obj,fig)
% fig Create separate figures according to one factor
%
% Example syntax : Gramm_object.fig(variable)
% For each unique value of variable, a new Gramm figure will be created,
% containing only the corresponding data. Useful when facet_ generates too
% crowded figures. Warning: the generated Gramm figures are independent: no
% common legend, axis limits, etc.

obj.aes.fig=shiftdim(fig);

end

