% gm1 = GMcondmarg(gm[,o]) Conditional or marginal Gaussian mixture
%
% Computes the parameters of a desired conditional or marginal Gaussian
% mixture from a joint Gaussian mixture.
%
% In:
%   gm,o: see GMpdf.m.
% Out:
%   gm1: GM mixture struct for the conditional or marginal.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function gm1 = GMcondmarg(gm,o)

% ---------- Argument defaults ----------
if ~exist('o','var') || isempty(o) gm1 = gm; return; end
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[M,D] = size(mu); I = o.P; J = o.M;

if length(I)==0				% Marginal
  if length(J)==0 || length(J)==D
    gm1 = gm;
  else
    gm1.p = pm; gm1.c = mu(:,J); gm1.type = cov_type;
    switch cov_type
     case 'F', gm1.S = S(J,J,:);
     case {'i','I'}, gm1.S = S;
     case 'd', gm1.S = S(J);
     case 'D', gm1.S = S(:,J);
    end
  end
else					% Conditional
  tI = o.xP;
  if length(I)+length(J)==D		% Direct
    gm2.c = mu(:,I); gm2.p = pm; gm2.type = cov_type;
    % NOTE: we don't check for underflow in the exponentials
    switch cov_type
     case 'F'
      gm2.S = S(I,I,:);
      for m=1:M
        mu1(m,:) = mu(m,J) + (S(I,J,m)'*(S(I,I,m)\(tI-mu(m,I))'))';
        S1(:,:,m) = S(J,J,m) - S(I,J,m)'*(S(I,I,m)\S(I,J,m));        
      end
     case {'i','I'}, gm2.S = S; mu1 = mu(:,J); S1 = S;
     case 'd', gm2.S = S(I); mu1 = mu(:,J); S1 = S(J);
     case 'D', gm2.S = S(:,I); mu1 = mu(:,J); S1 = S(:,J);
    end
    [~,~,pm_tI] = GMpdf(tI,gm2);
    gm1.c = mu1; gm1.S = S1; gm1.p = pm_tI'; gm1.type = cov_type;
  else					% Indirect
    % Do marginal
    o2.P = []; o2.M = [J I]; o2.xP = o.xP; gm2 = GMcondmarg(gm,o2);
    % Reorder
    [~,o1.P] = intersect(o2.M,I); [~,o1.M] = intersect(o2.M,J);
    o1.xP = o.xP;
    % Do conditional
    gm1 = GMcondmarg(gm2,o1);
  end
end

