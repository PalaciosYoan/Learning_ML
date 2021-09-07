% [p,g,H] = GMgradhess(x,gm[,o]) Gradient and Hessian of a Gaussian mixture
%
% Computes the value, gradient and Hessian of a Gaussian mixture at point x.
%
% In:
%   x: 1xD vector.
%      NOTE: this must have D columns even if o.M has fewer than D variables.
%   gm,o: see GMpdf.m.
% Out:
%   p: real number containing the density value, p(x).
%   g: 1xD vector containing the gradient at x.
%   H: DxD matrix containing the Hessian at x.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [p,g,H] = GMgradhess(x,gm,o)

% ---------- Argument defaults ----------
if exist('o','var') && ~isempty(o)
  % Transform parameters, then call the function again without "o"
  switch nargout
   case 1, p = GMgradhess(x(:,o.M),GMcondmarg(gm,o));
   case 2, [p,g] = GMgradhess(x(:,o.M),GMcondmarg(gm,o));
   otherwise, [p,g,H] = GMgradhess(x(:,o.M),GMcondmarg(gm,o));
  end     
  return;
end
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[M,D] = size(mu);

[p,~,~,pxm] = GMpdf(x,gm); g = zeros(D,1); H = zeros(D,D);

switch cov_type
 case 'F'
  for m = 1:M
    S_inv(:,:,m) = inv(S(:,:,m));
    tt = S_inv(:,:,m)*(mu(m,:)-x)';
    g = g + pxm(m)*tt; H = H + pxm(m)*(tt*tt'-S_inv(:,:,m));
  end
 case 'D'
  for m=1:M
    tt = diag(1./S(m,:))*(mu(m,:)-x)';
    g = g + pxm(m)*tt; H = H + pxm(m)*(tt*tt'-diag(sparse(1./S(m,:))));
  end
 case 'd'
  S_inv = diag(1./S);
  for m=1:M
    tt = S_inv*(mu(m,:)-x)';
    g = g + pxm(m)*tt; H = H + pxm(m)*(tt*tt'-S_inv);
  end
 case 'I'
  for m=1:M
    tt = (speye(D,D)/S(m))*(mu(m,:)-x)';
    g = g + pxm(m)*tt; H = H + pxm(m)*(tt*tt'-speye(D,D)/S(m));
  end
 case 'i'
  mux = bsxfun(@minus,mu,x);
  g = mux'*pxm'/S; H = mux'*diag(sparse(pxm))*mux/S^2-sum(pxm,2)*speye(D,D)/S;
end

g = g';

