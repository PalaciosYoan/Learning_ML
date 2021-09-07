% [Mean,C,pMean] = GMmoments(gm[,o]) Gaussian mixture mean and covariance
%
% In:
%   gm,o: see GMpdf.m.
% Out:
%   Mean: 1xD vector containing the mean.
%   C: DxD symmetric positive definite matrix containing the covariance.
%   pMean: real number containing the value of p(x) at the mean.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [Mean,C,pMean] = GMmoments(gm,o)

% ---------- Argument defaults ----------
if exist('o','var') && ~isempty(o)
  % Transform parameters, then call the function again without "o"
  switch nargout
   case 1, Mean = GMmoments(GMcondmarg(gm,o));
   case 2, [Mean,C] = GMmoments(GMcondmarg(gm,o));
   otherwise, [Mean,C,pMean] = GMmoments(GMcondmarg(gm,o));
  end
  return;
end
% ---------- End of "argument defaults" ----------

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;	% Extract GM fields
[M,D] = size(mu);

Mean = pm'*mu;

if nargout>=2

  C = - Mean'*Mean;

  switch cov_type
   case 'F', C = C + reshape(reshape(S,D*D,M)*pm,D,D);
   case 'i', C = C + diag(repmat(S,D,1));
   case 'I', C = C + diag(repmat(pm'*S,D,1));
   case 'd', C = C + diag(S);
   case 'D', C = C + diag(pm'*S);
  end
  
  if (D<10) && (M>1000)			% Vectorisation is faster
    % Process by blocks to avoid running out of memory.
    % Block size, select as large as possible; this will fit in mem GB RAM:
    mem = 1; B = floor((mem*1024^3)/(4*D*D*8)/2);
    i1 = 1; i2 = min(M,B);
    ind=repmat(1:D,D,1); ind1=ind(:); ind=ind'; ind2=ind(:);
    while i1 <= M
      C = C + reshape(pm(i1:i2)'*(mu(i1:i2,ind1).*mu(i1:i2,ind2)),D,D);
      i1 = i1 + B; i2 = min(M,i1+B-1);
    end
  else					% Loop over m=1:M is faster
    for m=1:M C = C + (pm(m)*mu(m,:)')*mu(m,:); end
  end
  
end

if nargout>2 pMean = GMpdf(Mean,gm); end;

