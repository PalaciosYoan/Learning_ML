rng(0);		% seed the random number generator for repeatability

% Parameters for a Gaussian mixture with 4 components in 2D
gm.p = [2 2 1 5]'; gm.p = gm.p/sum(gm.p); gm.c = [0 1;-2 5;-3 4;7 5];
gm.S(:,:,1) = [1 0;0 2]; gm.S(:,:,2) = [2 1;1 1];
gm.S(:,:,3) = [0.2 0.1;0.1 1]; gm.S(:,:,4) = [3 -1;-1 2]; gm.type = 'F';

% Range of the variables: [-5 10]x[0 7]
x1 = linspace(-5,10,200)'; x2 = linspace(0,7,200)';
[X1,X2] = meshgrid(x1,x2); X = [X1(:) X2(:)];

% Compute 2D pdf p(x) of x = (x1,x2) with GMpdf
[p,px_m,pm_x,pxm] = GMpdf(X,gm);

% Compute 1D conditional pdf p(x1|x2=6.8) with GMpdf
o1_2.P = 2; o1_2.M = 1; o1_2.xP = 6.8; p1_2 = GMpdf([x1 x1],gm,[],o1_2);
% Its parameters, explicitly computed with GMcondmarg
gm1_2 = GMcondmarg(gm,o1_2); gm1_2.p,gm1_2.c,gm1_2.S

% Compute 1D marginal pdf p(x1) with GMpdf
o1.P = []; o1.M = 1; p1 = GMpdf([x1 x1],gm,[],o1);

% Find modes of each pdf with GMmodes
modes = GMmodes(gm);
[modes1_2,pmodes1_2] = GMmodes(gm,o1_2);
[modes1,pmodes1] = GMmodes(gm,o1);

% Compute moments with GMmoments
[M,C] = GMmoments(gm)
[M,C] = GMmoments(gm,o1_2)

% Compute density, gradient and Hessian with GMgradhess at some points
[f,g,H] = GMgradhess(modes(1,:),gm)
[f,g,H] = GMgradhess([8 6],gm)
[f,g,H] = GMgradhess(2.7,gm,o1)

% Sample 350 points in 2D with GMsample
N = 350; Y = GMsample(N,gm);

% Estimate GM parameters from sample with GMEM
gm_EM = GMEM(Y,length(gm.p),'F',[],1e-7); gm_EM.c,gm_EM.S,gm_EM.p

% Estimate a kernel density estimate (KDE) from sample with bandwidth S = 0.5
gm_kde.p = ones(N,1)/N; gm_kde.c = Y; gm_kde.S = 0.5; gm_kde.type = 'i';
p_kde = GMpdf(X,gm_kde);


% Plot results in 2D and 1D

% Original GM in 2D
figure(1); contour(X1,X2,reshape(p,size(X1)),30);
hold on; plot(gm.c(:,1),gm.c(:,2),'k+',modes(:,1),modes(:,2),'ko'); hold off;
daspect([1 1 1]);

% Conditional GM in 1D
figure(2);
plot(x1,p1_2,'b-',x1,p1,'r-',modes1_2,pmodes1_2,'ko',modes1,pmodes1,'ko');

% Sample and estimated GM in 2D
figure(3); plot(Y(:,1),Y(:,2),'k*',gm_EM.c(:,1),gm_EM.c(:,2),'r+');
text(gm_EM.c(:,1),gm_EM.c(:,2),num2str([1:length(gm_EM.p)]'),'FontSize',24,...
     'Color','r');
daspect([1 1 1]);
% Plot covariance matrix for each component as one standard deviation contour
t = 0:2*pi/1000:2*pi; xy = [cos(t)' sin(t)']; xx = [0 1;-1 0;0 -1;1 0];
hold on;
for m = 1:length(gm_EM.p)
  switch gm_EM.type
   case 'F', [V,D] = eig(gm_EM.S(:,:,m));
   case 'D', [V,D] = eig(diag(gm_EM.S(m,:)));
   case 'd', [V,D] = eig(diag(gm_EM.S));
   case 'I', [V,D] = eig(gm_EM.S(m)*eye(size(Y,2)));
   case 'i', [V,D] = eig(gm_EM.S*eye(size(Y,2)));
  end
  mu = gm_EM.c(m,:);
  xy_new = bsxfun(@plus,xy*(V*sqrt(D))',mu);
  yy = bsxfun(@plus,xx*(V*sqrt(D))',mu);
  plot([yy(1,1) yy(3,1)],[yy(1,2) yy(3,2)],'r-',...
       [yy(2,1) yy(4,1)],[yy(2,2) yy(4,2)],'r-',...
       xy_new(:,1),xy_new(:,2),'r-','LineWidth',2);
end
hold off;

% KDE in 2D
figure(4); contour(X1,X2,reshape(p_kde,size(X1)),30);
daspect([1 1 1]);

