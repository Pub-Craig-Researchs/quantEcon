function vardecomp = VarDecomp(nvars, nsteps, ir)
%VARDECOMP Variance decomposition for impulse responses

arguments
    nvars (1,1) double {mustBeInteger, mustBePositive}
    nsteps (1,1) double {mustBeInteger, mustBePositive}
    ir (:,:,:) double
end

rng(0, "twister");

resp6 = zeros(nvars, nvars, nsteps);
resp7 = zeros(nsteps, 1);
vardecomp = zeros(nvars, nvars, nsteps);

for i = 1:nvars
    for j = 1:nvars
        resp = squeeze(ir(i, j, :));
        vardeco = resp .* resp;
        resp6(i, j, :) = cumsum(vardeco);
    end

    for k = 1:nsteps
        temp = resp6(i, :, k);
        resp7(k, 1) = sum(temp);
    end
    resp8 = resp7';

    for j = 1:nvars
        vardecomp(i, j, :) = squeeze(resp6(i, j, :))' ./ resp8;
    end
end
end
