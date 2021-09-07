% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

% Generate Swissroll dataset
rng(1);						% Fix random number gen. seed
N = 6000;					% Number of training points
Nt = 2000;					% Number of test points
Nall = N + Nt;
a = ((7/2*pi-pi/2)*(rand(Nall,1).^0.65)+pi/2);
Xall = [2+a.*cos(a) 1.5+a.*sin(a) 100*rand(Nall,1)];
X = Xall(1:N,:); C = a(1:N);			% Training dataset
Xt = Xall(N+1:end,:); Ct = a(N+1:end);		% Test dataset

%save swissroll X C Xt Ct

% Plot Swissroll
figure(1); scatter3(X(:,1),X(:,2),X(:,3),3,C);
axis([-10 10 -10 10 0 100]); daspect([1 1 1]);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');

% Fly around to obtain different views
fr = 40;
el = linspace(-90,90,fr); el = [el el([end-1:-1:2])];
az = 30+linspace(0,360,length(el)+1); az = az([1:end-1]);
for k=1:length(az)
  view(az(k),el(k)); title(['az=' num2str(az(k),3) '; el=' num2str(el(k),3)]);
  pause(0.3);
end

