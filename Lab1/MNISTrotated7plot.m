% Copyright (c) 2016 by Miguel A. Carreira-Perpinan
% for use in CSE176 Introduction to Machine Learning at UC Merced

load MNISTrotated7.mat; D = size(train_img,2); DD = sqrt(D);

N = 50; M = 60;

% Plot some (image,skeleton) pairs
figure(1);
for i=3:10:N
  for j=1:M
    k = (i-1)*M+j;
    clf;
    subplot(1,2,1);
    imagesc(reshape(train_img(k,:),DD,DD)'); colormap(gray(256)); axis image;
    title(['image ' num2str(i)]);
    subplot(1,2,2);
    tmp = reshape(train_skel(k,:)',[size(train_skel,2)/2 2]);
    hold on;
    plot(tmp(:,1),tmp(:,2),'ro','MarkerFaceColor','r');
    plot(tmp(1:11,1),tmp(1:11,2),'k-');
    plot(tmp([12 13 7 14],1),tmp([12 13 7 14],2),'k-');
    axis([-5 5 -5 5]); set(gca,'DataAspectRatio',[1 1 1]); box on;
    title(['angle = ' num2str(train_angle(k)*180/pi)]);
    hold off;
    pause;
  end
end

