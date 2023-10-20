%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
N = 101;

seed = rand*1000;
rng(824);

t = linspace(0,1,N);
f1 = @(x) 6*(0.8).^(20*x).*cos(10*pi.*x-pi/4);
f2 = @(x) 5.5*(0.8).^(20*x).*sin(10*pi.*x);
time_gap = 1/(N-1);
a = [-0.5, 2];
f1 = f1(t)';


% % random generate true gamma2 to align f to get f2
% tan2 = zeros(1, N);
% 
% S = [];
% for j =1: dn
%     S(j) = normrnd(0, sigma/j);
%     tan2 = tan2 + S(j)*b1(j,:);
% end
% tan2 = tan2-log(trapz(t,exp(tan2)));
% 
% temp = exp(tan2)./trapz(t, exp(tan2),2);
% gamma2 = cumsum(temp.^2,2)/sum(temp.^2,2);    

% % set fixed gamma2
gamma2 = (exp(a(2)*t)-1)/(exp(a(2))-1);
gamma_true = interp1(gamma2, t, t);
f2 = 1.1*interp1(t,f2(t),gamma2);
f2 = f2';
figure(1); clf;
plot(t, f1, t, f2);
q1 = sign(gradient(f1)/time_gap).*sqrt(abs(gradient(f1)/time_gap));
q2 = sign(gradient(f2)/time_gap).*sqrt(abs(gradient(f2)/time_gap));

%define the covariance for the gamma function

% f_cov = ones(1, N);
% % f_cov= [5*ones(1, (N-1)/2+31), 0.1*ones(1, (N-1)/2-30)];
% Cr1 = diag(f_cov);
% sigma_kernel = 10;
% kernel_size = 51; % Adjust the size as needed
% [X, Y] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2, -(kernel_size-1)/2:(kernel_size-1)/2);
% gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma_kernel^2));
% Cr2 = 0.08*conv2(Cr1, gaussian_kernel, 'same'); %change to 0.01, we can get second-optimal
% Cr = Cr2;
% [V, D, U] = svd(Cr);
% neg_eigenvalues = find(diag(D) < 0);
% D(neg_eigenvalues, neg_eigenvalues) = D(neg_eigenvalues, neg_eigenvalues)*-1;
% Cr = V * D * V';



% Define the value of pho
pho = 0.999;  % You can change this to your desired value
% Initialize an empty matrix of size (n+1) x (n+1)

f_cov = ones(1, N);
% f_cov= [ones(1, (N-1)/2+1), 0.99*ones(1, (N-1)/2)];
Cr = diag(f_cov);
% Cr = zeros(N, N);

% Fill in the matrix
for i = 1:N
    for j = 1:N
        if i ~= j
            Cr(i, j) = pho^(abs(i-j));
        end
    end
end

Cr = 5*Cr;
mu = zeros(1, N);

% figure(8);clf
% EX = mvnrnd(mu, Cr, 50);
% plot(t, EX)
figure(1001); clf;
imagesc(Cr);
try chol(Cr)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

inv_cr = inv(Cr);
figure(1002); clf;
imagesc(inv_cr);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dynamic programming to match f1 and f2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma_t = DynamicProgrammingQ_Adam(q2', q1', 0, 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bayesian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. initiate phi 
% svd of Cr
[U, S, V] = svd(Cr);
S = S*time_gap;

dn = 50;
for j = 1:dn
    U1(j,:) = U(:,j);
    U1(j,:) = U1(j,:)/sqrt(trapz(t, U1(j,:).^2)); 
end

phi_int = zeros(1, N);

% for j = 1:dn
%     phi_int = phi_int + normrnd(0, sqrt(S(j,j)))*U1(j,:);
% end
% phi_int = phi_int-log(trapz(t,exp(phi_int)));

phi_int = mvnrnd(mu, Cr, 1);
phi_int = phi_int-log(trapz(t,exp(phi_int)));

figure(101); clf;
plot(t, phi_int);
figure(102); clf;
gamma_in = cumsum(exp(phi_int))./sum(exp(phi_int));
gamma_in = (gamma_in-min(gamma_in))/(max(gamma_in)-min(gamma_in)); 
plot(t, gamma_in);
%%
%2. Initiate sigma1
sigma1_int = 2;

%3: update g and sigma1
J = 10000;
phi_set = [];
phi_set(1,:) = phi_int;

%set the parameters for the pCN-mixture
betals = [0.5, linspace(0.01, 0.1,9)];
probabilities = repmat(0.1, 1, 10);


for j = 1: J
    %propose new phi
    kesi = mvnrnd(mu, Cr, 1);
%     kesi = zeros(1, N);
%     for jj = 1:dn
%         kesi = kesi + normrnd(0, sqrt(S(jj,jj)))*U1(jj,:);
%     end
    kesi = kesi-log(trapz(t,exp(kesi)));
    beta = randsample(betals, 1,true, probabilities);
    phi_new = phi_set(j,:)*sqrt(1-beta^2) + beta*kesi;
    phi_new = phi_new-log(trapz(t,exp(phi_new)));
    
    % calculate MCMC acceptance ratio
    [lossratio, sse_diff(j)] = cal_joint_ratio_clr(sigma1_int, q1, q2, t, phi_new, phi_set(j,:));
    lossratio_(j) = lossratio;
    lamd_p = min(1, lossratio);

    if rand()<lamd_p
        phi_set(j+1,:) = phi_new;
    else
        phi_set(j+1,:) = phi_set(j,:);
    end
    
    temp_t = cumtrapz(t,exp(phi_set(j+1,:)))./trapz(t, exp(phi_set(j+1,:)),2);
    temp_t = round(temp_t/temp_t(end)*(N-1))+1;
    gam1_dev = exp(phi_set(j+1,:));
    SSE = (norm(q2 - q1(temp_t).*sqrt(gam1_dev')))^2;
    sse_(j) = SSE;

end

% plot the sample gamma
gamma_new = [];
cnt = 1;
for j = 1:J+1
%     phi_set(j,:) = phi_set(j,:)-log(trapz(t,exp(phi_set(j,:))));
    gamma_new(j,:) = cumsum(exp(phi_set(j,:)))./sum(exp(phi_set(j,:)));
    gamma_new(j,:) = (gamma_new(j,:)-min(gamma_new(j,:)))/(max(gamma_new(j,:))-min(gamma_new(j,:)));
end


fgamma = gamma_new(J/2+1:10:J+1,:);
fphi_set = phi_set(J/2+1:10:J+1,:);

eps = 1e-5;
learn_rate = 0.1;
sample_mean = mean(phi_set,1);
gamma_mean = cumsum(exp(sample_mean))./sum(exp(sample_mean));
gamma_mean = (gamma_mean-min(gamma_mean))/(max(gamma_mean)-min(gamma_mean));





figure(1); clf;
hold on;
plot(t,gamma_new);
plot(t,gamma_true ,'k','LineWidth',2);
plot(t,gamma_t,'m--','LineWidth',2);
plot(t,gamma_mean','b--','LineWidth',2);
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width      = 12;
opts.height     = 10;
opts.fontType   = 'Times';
% 
figure(4); clf;
hold on;
plot(t,fphi_set);


f2_gamma_t = interp1(t,f2,gamma_mean);
figure(2);clf;
plot(t,f1,'b.','LineWidth', 1.5);
hold on;
plot(t,f2,'g.','LineWidth', 1.5);
plot(t, f2_gamma_t, 'r','LineWidth', 1.5)

figure(3);clf;
plot(t,q1,'b.','LineWidth', 1.5);
hold on;
plot(t,q2,'g.','LineWidth', 1.5);

figure(5);clf;
subplot(2,1,1);
plot(sse_)
subplot(2,1,2);
hist(sse_);
% 
figure (8);clf;
for i =1:N
    eigenvalue(i) = S(i,i);
end
plot(cumsum(eigenvalue(1:50))/sum(eigenvalue),'linewidth', 1.5);


%% Perform k-means clustering
[idx, centers] = kmeans(gamma_new, 2);


figure(11); clf;
hold on;
plot(t,gamma_new(idx==1,:),'Color', [0.8 0.8 0.8]);
plot(t,gamma_new(idx==2,:),'g');
plot(t, centers(1,:),'k','LineWidth',2)
plot(t, centers(2,:),'b','LineWidth',2)
plot(t,gamma_t,'m--','LineWidth',2);
lsize = 16; % Label fontsize
nsize = 18; % Axis fontsize
axis equal;
ylim([0,1]);
xlim([0,1]);
xticks([0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Fontsize', nsize,'linewidth', 1.5)
set(gcf,'paperpositionmode','auto');
set(gcf,'windowstyle','normal');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gca,'fontweight','normal')
opts.Colors     = get(groot,'defaultAxesColorOrder');
opts.saveFolder = 'img/';
opts.width      = 12;
opts.height     = 10;
opts.fontType   = 'Times';

f2_gamma_t1 = interp1(t,f2,centers(1,:));
f2_gamma_t2 = interp1(t,f2,centers(2,:));
figure(22);clf;
plot(t,f1,'b.','LineWidth', 1.5);
hold on;
plot(t,f2,'g.','LineWidth', 1.5);
plot(t, f2_gamma_t1, 'k','LineWidth', 1.5)
plot(t, f2_gamma_t2, 'b','LineWidth', 1.5)