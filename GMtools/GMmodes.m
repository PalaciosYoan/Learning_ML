% [modes,pmodes,Hessians,codes,its,labels] =
%   GMmodes(gm[,o,tol,maxit,mindiff,maxeig,th])
% Gaussian mixture mode-finding by fixed-point (mean-shift) iteration
%
% GMmodes finds all the modes of a Gaussian mixture (GM), defined as:
%    p(x) = \sum^M_{m=1}{p(m) p(x|m)}
% where p(x|m) is a Gaussian distribution of mean mu(m) and covariance
% matrix S.
%
% GMmodes searches for modes starting from every Gaussian centroid (contained
% rowwise in the matrix mu). This procedure should typically find all the
% modes. Repeated modes (due to the coalescence of several Gaussian
% components) are removed. 
%
% The mode search is done by a simple fixed-point iterative algorithm,
% which needs no ad-hoc parameters to control convergence (e.g., step size).
%
% If we consider the GM as a kernel density estimate defined on M data points,
% the labels returned (which associate each point with a mode) achieve
% "mean-shift clustering".
%
% Notes:
% - Running the mode-finding algorithm from each of the M centroids results
%   in M points. Some of these, although numerically different, should
%   correspond to the same mode of the GM. This can be solved by aggregating
%   points that are within a distance "mindiff". The code below does this as
%   soon as each centroid is processed ("update mode list"). It is also
%   possible to process all centroids and then run a connected-components
%   algorithm with threshold "mindiff".
% - Confidence intervals around the modes found can be computed based on the
%   Hessian at each mode, as described in the references below.
%
% References:
% - Miguel A. Carreira-Perpinan (2000): "Mode-finding for mixtures of
%   Gaussian distributions", IEEE Trans. on Pattern Analysis and
%   Machine Intelligence 22(11): 1318-1323.
% - Miguel A. Carreira-Perpinan (1999): "Mode-finding for mixtures of
%   Gaussian distributions", Technical report CS-99-03, Dept. of
%   Computer Science, University of Sheffield, UK (revised Aug. 4, 2000).
% - Miguel A. Carreira-Perpinan (2015): "Clustering methods based on kernel
%   density estimators: mean-shift algorithms". Handbook of Cluster Analysis,
%   Chapman \& Hall/CRC.
%
% In:
%   gm,o: see GMpdf.m.
%   tol: minimum distance between successive points to keep iterating
%      (default 1e-4 standard deviations).
%   maxit: maximum number of iterations for the optimisation algorithm.
%      Default: 1000.
%   mindiff: minimum distance between modes to be considered the same. Note
%      that "tol" has to be small enough so that the iterative procedure does
%      not stop so early that the new mode will not be considered different
%      from the previously found ones. Default: 1e-2 standard deviations.
%   maxeig: maximum algebraic value for an eigenvalue of the Hessian to be
%      considered definite negative. It is only used when deciding whether to
%      keep or throw away a fixed-point point, not during the maximisation
%      itself. Default: 0.
%   th: threshold in [0,1). Components whose mixing proportion is smaller or
%      equal than th*(largest mixing proportion) are discarded. Default: 0.01.
% Out:
%   modes: KxD matrix containing the K distinct modes rowwise. The modes are
%      returned sorted by descending probability (pmodes).
%   pmodes: Kx1 list containing the value of p(x) at each mode.
%   Hessians: DxDxK array. Hessians(:,:,k) is the Hessian at the kth mode.
%   codes: Kx1 list containing the exit code of the maximisation algorithm
%      (0: near-zero gradient found; 1: maximum number of iterations reached).
%   its: Kx1 list containing the number of iterations employed in the
%      maximisation algorithm.
%   labels: Mx1 list containing the index in "modes" corresponding to the
%      starting GM centroid (i.e., it points to the mode of convergence).
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% Copyright (c) 2011 by Miguel A. Carreira-Perpinan and Chao Qin

function [modes,pmodes,Hessians,codes,its,labels] = ...
      GMmodes(gm,o,tol,maxit,mindiff,maxeig,th)

mu = gm.c; S = gm.S; pm = gm.p; cov_type = gm.type;  % Extract GM fields
[M,D] = size(mu);

% ---------- Argument defaults ----------
if ~exist('tol','var') || isempty(tol) tol = []; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 1000; end;
if ~exist('mindiff','var') || isempty(mindiff) mindiff = []; end;
if ~exist('maxeig','var') || isempty(maxeig) maxeig = 0; end;
if ~exist('th','var') || isempty(th) th = 1/100; end;
if exist('o','var') && ~isempty(o)
  % Transform parameters, then call the function again without "o"
  switch nargout
   case 1, modes = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   case 2, [modes,pmodes] = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   case 3, [modes,pmodes,Hessians] = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   case 4, [modes,pmodes,Hessians,codes] = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   case 5, [modes,pmodes,Hessians,codes,its] = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   otherwise, [modes,pmodes,Hessians,codes,its,labels] = ...
    GMmodes(GMcondmarg(gm,o),[],tol,maxit,mindiff,maxeig,th); 
   end
  return;
end
if isempty(tol) || isempty(mindiff)
  switch cov_type
   case 'F', for m=1:M mineig(m) = min(eig(S(:,:,m))); end; v = min(mineig);
   case {'i','I','d','D'}, v = min(S(:));
  end
  if isempty(tol) tol = 1e-4*sqrt(v); end;
  if isempty(mindiff) mindiff = 1e-2*sqrt(v); end;
end
% ---------- End of "argument defaults" ----------

% Discard all components whose mixing proportion is smaller than a threshold
% (1% of the top value). This will speed up the search and probably won't miss
% any mode.
z = find(pm>max(pm)*th); pm = pm(z); pm = pm/sum(pm); mu = mu(z,:);
switch cov_type
 case 'F', S = S(:,:,z);
 case 'D', S = S(z,:);
 case 'I', S = S(z); 
end
M = size(mu,1);					% New number of components

% For mixtures of one component, the calculation is direct.
if M==1
  modes = mu; pmodes = GMpdf(modes,gm);
  if nargout >= 3
    codes = 0; its = 0; Hessians = zeros(D,D,1); labels = 1;
    switch cov_type
     case 'F', Hessians(:,:,1) = -pmodes*inv(S);
     case {'D','d'}, Hessians(:,:,1) = -pmodes*diag(1./S);
     case {'d','I'}, Hessians(:,:,1) = -pmodes*eye(D,D)/S;
    end
  end
  return;
end
gm.c = mu; gm.S = S; gm.p = pm;
modes = [];
terminal = zeros(size(mu));

for m=1:M	% Perform iterative procedure from every centroid in mu

  x = mu(m,:);					% Starting point: mth centroid
  x_old = x;
  code = -1; it = 0;
  
  while code < 0	                  	% Fixed-point loop
    switch cov_type
     case 'F'
      [p,~,pm_x,pxm] = GMpdf(x,gm);
      for k=1:M
        S_inv(:,:,k) = inv(S(:,:,k));
        tt1(:,:,k) = pm_x(k)*S_inv(:,:,k);
        tt2(k,:) = tt1(:,:,k)*mu(k,:)';
      end
      tt3 = sum(tt1,3);
      tt4 = sum(tt2,1);
      x = (tt3\tt4')';			       % Fixed-point iteration
     case 'D'
      [p,~,pm_x,pxm] = GMpdf(x,gm);
      tt1 = bsxfun(@rdivide,pm_x',S);
      tt2 = sum(tt1.*mu,1);
      tt3 = diag(1./sum(tt1,1));
      x = tt2*tt3';
     case {'d','i'}
      [p,~,pm_x,pxm] = GMpdf(x,gm);
      x = pm_x*mu;
     case 'I'
      [p,~,pm_x,pxm] = GMpdf(x,gm);      
      qxm = pm_x./S';
      qm_x = qxm/sum(qxm,2);      
      x = qm_x*mu;
    end
    
    if it>=maxit				% Max. no. iterations reached
      code = 1;
    elseif norm(x-x_old) < tol			% Tolerance achieved
      code = 0;
    else
      x_old = x;				% Keep current point
      it = it + 1;				% Continue iterating
    end
  end
  [~,g,H] = GMgradhess(x,gm);
  terminal(m,:) = x;				% Final iterate from mu(m,:)

  % ---------- Update mode list ----------
  % To avoid adding points with nonnegative Hessian (saddle or minimum
  % points, usually due to areas of very low probability),
  % we ensure that the maximum eigenvalue (in algebraic value) is not
  % positive as indicated by maxeig.
  % Then we add (x,px,H) to the list if x is new, or if it was
  % already in the list but this one has higher probability, we
  % replace the old one(s). Note that it is necessary to replace not
  % just the old x in the list which is closest to the new one, but
  % we need to check every single old x in the list. For example, if
  % mindiff=0.1 and modes=[1.92 2.04] and x=2.01, the new list
  % should be modes=[2.01] and not [1.92 2.04].
  if max(eig(H))<maxeig				% H<0 ?
    if isempty(modes)				% First mode found?
      modes = x;
      pmodes = sum(pxm,2);
      if nargout>=3
        Hessians = zeros(D,D,1); Hessians(:,:,1) = H;
      end
      if nargout>=4
        codes = code;
      end;
      if nargout>=5
        its = it;
      end
    else                                % models found already
      temp1 = sqrt(sum((modes-x(ones(length(modes(:,1)),1),:)).^2,2))>mindiff;
      same = find(temp1==0);		% These are all equivalent to x
      not_same = find(temp1==1);	% These are all different from x
      % Determine the old x with maximum probability
      if isempty(same)
        temp1 = -1;
      else
        [temp1,temp2] = max(pmodes(same));
      end
      % Update modes, pmodes, Hessians, codes and its
      if temp1>=sum(pxm,2)				% Don't add the new x
        keep = [not_same; same(temp2)];
        modes = modes(keep,:);				% Update modes
        pmodes = pmodes(keep);				% Update pmodes
        if nargout>=3
          Hessians = Hessians(:,:,keep);
        end;						% Update Hessians
        if nargout>=4
          codes = codes(keep);
        end;						% Update codes
        if nargout>=5
          its = its(keep);
        end;						% Update its
      else						% Add the new x
        modes = [modes(not_same,:); x];			% Update modes
        pmodes = [pmodes(not_same); sum(pxm,2)];	% Update pmodes
        if nargout>=3       				% Update Hessians
          temp1 = zeros(D,D,length(pmodes));
          if ~isempty(not_same)
            temp1(:,:,1:length(pmodes)-1) = Hessians(:,:,not_same);
          end
          temp1(:,:,length(pmodes)) = H;
          Hessians = temp1;
        end
        if nargout>=4
          codes = [codes(not_same);code];
        end;						% Update codes
        if nargout>=5
          its = [its(not_same); it];
        end;						% Update its
      end
    end
  end
  % ---------- End of "update mode list" -------------
end

% ---------- Sort modes by descending probability ----------
[~,temp2] = sort(-pmodes); modes = modes(temp2,:); pmodes = pmodes(temp2);
if nargout>=3 Hessians = Hessians(:,:,temp2); end;
if nargout>=4 codes = codes(temp2); end;
if nargout>=5 its = its(temp2); end;
% ---------- End of "sort modes by descending probability" ----------

% Assign each centroid to a mode (label).
% Rather than keeping the labels all the way and update, merge, etc. them as
% modes are updated (which would be complicated) we simply assign each point
% in mu to the mode that is closest to its terminal point in paths.
if nargout > 5
  [~,labels] = min(sqdist(terminal,modes),[],2);
end

