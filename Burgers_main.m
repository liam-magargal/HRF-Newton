clc;
clear all;
close all;

% user inputs
N = 1024;
dt = .001;
Nt = 500;

mu_left = 3.125;
mu2 = .0175;

% mu_left = 1.375;
% mu2 = .0825;

Nh = 5000; % number of sample solutions used to get the ECSW weights (following the procedure of Grimberg, 2020)

dx = 1/(N+1);
x = linspace(dx,1-dx,N);
x_hist = zeros(N,Nt);
tol = 1e-6;

t_domain = linspace(0,dt*Nt,Nt);
[X, T] = meshgrid(x,t_domain);


[C, A, F, B, M] = getCoeffMat(N,x,dx);

x_hist = getAllTrainingSolutionsOptimized(N,dt,Nt,tol,x,dx);


[x_hist_test, FOMtime] = getSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x);
FOMtime



%% ROMs
[U,Sigma,V] = svd(x_hist,'econ');

latTol = [10^(-2/2) 10^(-3/2) 10^(-4/2) 10^(-5/2) 10^(-6/2) 10^(-7/2) 10^(-8/2)]';

uStan_error = zeros(N,1);

for i=1:N
    uStan_error(i) = 1-sum(diag(Sigma(1:i,1:i)).^2) / sum(diag(Sigma).^2);
end

reconError = zeros(size(latTol,1),1);


statePredictionErrorStandardLSPG = zeros(size(latTol,1),1);
statePredictionErrorStandardGalerkin = zeros(size(latTol,1),1);
statePredictionErrorHFLSPG = zeros(size(latTol,1),1);
statePredictionErrorHFGalerkin = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e5 = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e7 = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e9 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e5 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e7 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e9 = zeros(size(latTol,1),1);


ROMevaluationErrorHFLSPG = zeros(size(latTol,1),1);
ROMevaluationErrorHFGalerkin = zeros(size(latTol,1),1);
ROMevaluationErrorECSWLSPG1e5 = zeros(size(latTol,1),1);
ROMevaluationErrorECSWLSPG1e7 = zeros(size(latTol,1),1);
ROMevaluationErrorECSWLSPG1e9 = zeros(size(latTol,1),1);
ROMevaluationErrorECSWGalerkin1e5 = zeros(size(latTol,1),1);
ROMevaluationErrorECSWGalerkin1e7 = zeros(size(latTol,1),1);
ROMevaluationErrorECSWGalerkin1e9 = zeros(size(latTol,1),1);


speedupFactorStandardGalerkin = zeros(size(latTol,1),1);
speedupFactorStandardLSPG = zeros(size(latTol,1),1);
speedupFactorHFLSPG = zeros(size(latTol,1),1);
speedupFactorHFGalerkin = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e5 = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e7 = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e9 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e5 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e7 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e9 = zeros(size(latTol,1),1);


noSampledPointsECSWLSPG1e5 = zeros(size(latTol,1),1);
noSampledPointsECSWLSPG1e7 = zeros(size(latTol,1),1);
noSampledPointsECSWLSPG1e9 = zeros(size(latTol,1),1);
noSampledPointsECSWGalerkin1e5 = zeros(size(latTol,1),1);
noSampledPointsECSWGalerkin1e7 = zeros(size(latTol,1),1);
noSampledPointsECSWGalerkin1e9 = zeros(size(latTol,1),1);




for i=1:size(latTol,1)
    i
    for j=1:size(uStan_error,1)
        if uStan_error(j)<latTol(i)
            numModes = j;
            break
        end
    end


    phi = U(:,1:numModes);

    x_hist_recon = phi*phi'*x_hist_test;
    reconError(i) = norm(x_hist_recon - x_hist_test,'fro')^2 / norm(x_hist_test,'fro')^2;

    
    [x_hat_Galerkin, ROMtime] = getStandardGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,phi);
    x_hist_Galerkin = phi*x_hat_Galerkin;
    statePredictionErrorStandardGalerkin(i) = norm(x_hist_Galerkin - x_hist_test,'fro')^2 / norm(x_hist_test,'fro')^2;
    speedupFactorStandardGalerkin(i) = FOMtime/ROMtime;


    [x_hat_LSPG, ROMtime] = getStandardLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,phi);
    x_hist_LSPG = phi*x_hat_LSPG;
    statePredictionErrorStandardLSPG(i) = norm(x_hist_LSPG - x_hist_test,'fro')^2 / norm(x_hist_test,'fro')^2;
    speedupFactorStandardLSPG(i) = FOMtime/ROMtime;


    [x_hat_HFGalerkin, ROMtime] = getSolutionHFGalerkinOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,C,A,F,B,M,phi);
    speedupFactorHFGalerkin(i) = FOMtime/ROMtime;
    x_approx_HFGalerkin = phi*x_hat_HFGalerkin;
    statePredictionErrorHFGalerkin(i) = norm(x_approx_HFGalerkin - x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorHFGalerkin(i) = norm(x_approx_HFGalerkin - x_hist_Galerkin,'fro')^2 / norm(x_hist_Galerkin,'fro')^2;


    [x_hat_HFLSPG, ROMtime] = getSolutionHFLSPGOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,C,A,F,B,M,phi);
    speedupFactorHFLSPG(i) = FOMtime/ROMtime;
    x_approx_HFLSPG = phi*x_hat_HFLSPG;
    statePredictionErrorHFLSPG(i) = norm(x_approx_HFLSPG - x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorHFLSPG(i) = norm(x_approx_HFLSPG - x_hist_LSPG,'fro')^2 / norm(x_hist_LSPG,'fro')^2;


    load(strcat('BurgersECSWweights/LSPG_1e5_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWLSPG1e5 = phi*x_hat_hist;
    statePredictionErrorECSWLSPG1e5(i) = norm(x_approx_ECSWLSPG1e5-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWLSPG1e5(i) = norm(x_approx_ECSWLSPG1e5-x_hist_LSPG,'fro')^2/norm(x_hist_LSPG,'fro')^2;
    noSampledPointsECSWLSPG1e5(i) = size(indices,1);
    speedupFactorECSWLSPG1e5(i) = FOMtime / ROMtime;

    load(strcat('BurgersECSWweights/Galerkin_1e5_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWGalerkin1e5 = phi*x_hat_hist;
    statePredictionErrorECSWGalerkin1e5(i) = norm(x_approx_ECSWGalerkin1e5-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWGalerkin1e5(i) = norm(x_approx_ECSWGalerkin1e5-x_hist_Galerkin,'fro')^2/norm(x_hist_Galerkin,'fro')^2;
    noSampledPointsECSWGalerkin1e5(i) = size(indices,1);
    speedupFactorECSWGalerkin1e5(i) = FOMtime / ROMtime;


    load(strcat('BurgersECSWweights/LSPG_1e7_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWLSPG1e7 = phi*x_hat_hist;
    statePredictionErrorECSWLSPG1e7(i) = norm(x_approx_ECSWLSPG1e7-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWLSPG1e7(i) = norm(x_approx_ECSWLSPG1e7-x_hist_LSPG,'fro')^2/norm(x_hist_LSPG,'fro')^2;
    noSampledPointsECSWLSPG1e7(i) = size(indices,1);
    speedupFactorECSWLSPG1e7(i) = FOMtime / ROMtime;


    load(strcat('BurgersECSWweights/Galerkin_1e7_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWGalerkin1e7 = phi*x_hat_hist;
    statePredictionErrorECSWGalerkin1e7(i) = norm(x_approx_ECSWGalerkin1e7-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWGalerkin1e7(i) = norm(x_approx_ECSWGalerkin1e7-x_hist_Galerkin,'fro')^2/norm(x_hist_Galerkin,'fro')^2;
    noSampledPointsECSWGalerkin1e7(i) = size(indices,1);
    speedupFactorECSWGalerkin1e7(i) = FOMtime / ROMtime;

    load(strcat('BurgersECSWweights/LSPG_1e9_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWLSPG1e9 = phi*x_hat_hist;
    statePredictionErrorECSWLSPG1e9(i) = norm(x_approx_ECSWLSPG1e9-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWLSPG1e9(i) = norm(x_approx_ECSWLSPG1e9-x_hist_LSPG,'fro')^2/norm(x_hist_LSPG,'fro')^2;
    noSampledPointsECSWLSPG1e9(i) = size(indices,1);
    speedupFactorECSWLSPG1e9(i) = FOMtime / ROMtime;
    

    load(strcat('BurgersECSWweights/Galerkin_1e9_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x);
    x_approx_ECSWGalerkin1e9 = phi*x_hat_hist;
    statePredictionErrorECSWGalerkin1e9(i) = norm(x_approx_ECSWGalerkin1e9-x_hist_test,'fro')^2/norm(x_hist_test,'fro')^2;
    ROMevaluationErrorECSWGalerkin1e9(i) = norm(x_approx_ECSWGalerkin1e9-x_hist_Galerkin,'fro')^2/norm(x_hist_Galerkin,'fro')^2;
    noSampledPointsECSWGalerkin1e9(i) = size(indices,1);
    speedupFactorECSWGalerkin1e9(i) = FOMtime / ROMtime;


    if latTol(i)==1e-4
        x_approx_HFGalerkin_out = x_approx_HFGalerkin;
        x_approx_HFLSPG_out = x_approx_HFLSPG;
        x_approx_ECSWGalerkin1e5_out = x_approx_ECSWGalerkin1e5;
        x_approx_ECSWGalerkin1e7_out = x_approx_ECSWGalerkin1e7;
        x_approx_ECSWGalerkin1e9_out = x_approx_ECSWGalerkin1e9;
        x_approx_ECSWLSPG1e5_out = x_approx_ECSWLSPG1e5;
        x_approx_ECSWLSPG1e7_out = x_approx_ECSWLSPG1e7;
        x_approx_ECSWLSPG1e9_out = x_approx_ECSWLSPG1e9;
    end
  
end


function [x_hist_sol,FOMtime] = getSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x)

x_hist_sol = zeros(N,Nt);

tic
for t=2:Nt
    x_curr = x_hist_sol(:,t-1);
    x_next = x_hist_sol(:,t-1);

    while true
        r = zeros(N,1);
        J = zeros(N,N);

        r(1) = x_next(1) - x_curr(1) + dt/(2*dx)*x_next(1)*(x_next(2) - mu_left) - dt*mu2/dx/dx*(x_next(2) - 2*x_next(1) + mu_left);
        r(N) = x_next(N) - x_curr(N) + dt/(2*dx)*x_next(N)*(0 - x_next(N-1)) - dt*mu2/dx/dx*(0 - 2*x_next(N) + x_next(N-1));
        r(2:N-1) = x_next(2:N-1) - x_curr(2:N-1) + dt/(2*dx)*x_next(2:N-1).*(x_next(3:N) - x_next(1:N-2)) - dt*mu2/dx/dx*(x_next(1:N-2) - 2*x_next(2:N-1) + x_next(3:N));
        
        J(1,1) = 1 + dt/2/dx*(x_next(2)-mu_left) + 2*dt*mu2/dx/dx;
        J(1,2) = dt/2/dx*x_next(1) - mu2*dt/dx/dx;

        J(N,N) = 1 + dt/2/dx*(0-x_next(N-1)) + 2*dt*mu2/dx/dx;
        J(N,N-1) = -dt/2/dx*x_next(N) - mu2*dt/dx/dx;

        for i=2:N-1
            J(i,i) = 1 + dt/2/dx*(x_next(i+1)-x_next(i-1)) + 2*dt*mu2/dx/dx;
            J(i,i-1) = -dt/2/dx*x_next(i) - mu2*dt/dx/dx;
            J(i,i+1) = dt/2/dx*x_next(i) - mu2*dt/dx/dx;
        end
        
        if norm(r)>=tol
            x_next = x_next - J\r;
        else
            x_hist_sol(:,t) = x_next;
            break
        end
    end
end
FOMtime = toc;

end



function [x_hat_hist,ROMtime] = getStandardGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);

tic;
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        x_curr = phi*x_hat_curr;
        x_next = phi*x_hat_next;
        r = zeros(N,1);
        J = zeros(N,N);

        r(1) = x_next(1) - x_curr(1) + dt/(2*dx)*x_next(1)*(x_next(2) - mu_left) - dt*mu2/dx/dx*(x_next(2) - 2*x_next(1) + mu_left);
        r(N) = x_next(N) - x_curr(N) + dt/(2*dx)*x_next(N)*(0 - x_next(N-1)) - dt*mu2/dx/dx*(0 - 2*x_next(N) + x_next(N-1));
        r(2:N-1) = x_next(2:N-1) - x_curr(2:N-1) + dt/(2*dx)*x_next(2:N-1).*(x_next(3:N) - x_next(1:N-2)) - dt*mu2/dx/dx*(x_next(1:N-2) - 2*x_next(2:N-1) + x_next(3:N));
        
        J(1,1) = 1 + dt/2/dx*(x_next(2)-mu_left) + 2*dt*mu2/dx/dx;
        J(1,2) = dt/2/dx*x_next(1) - mu2*dt/dx/dx;

        J(N,N) = 1 + dt/2/dx*(0-x_next(N-1)) + 2*dt*mu2/dx/dx;
        J(N,N-1) = -dt/2/dx*x_next(N) - mu2*dt/dx/dx;

        for i=2:N-1
            J(i,i) = 1 + dt/2/dx*(x_next(i+1)-x_next(i-1)) + 2*dt*mu2/dx/dx;
            J(i,i-1) = -dt/2/dx*x_next(i) - mu2*dt/dx/dx;
            J(i,i+1) = dt/2/dx*x_next(i) - mu2*dt/dx/dx;
        end

        LHS = phi'*J*phi;
        RHS = phi'*r;
        
        
        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
    end
end
ROMtime = toc;

end




function [x_hat_hist,ROMtime] = getStandardLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);

tic;
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    iterCount = 1;
    while true
        x_curr = phi*x_hat_curr;
        x_next = phi*x_hat_next;
        r = zeros(N,1);
        J = zeros(N,N);

        r(1) = x_next(1) - x_curr(1) + dt/(2*dx)*x_next(1)*(x_next(2) - mu_left) - dt*mu2/dx/dx*(x_next(2) - 2*x_next(1) + mu_left);
        r(N) = x_next(N) - x_curr(N) + dt/(2*dx)*x_next(N)*(0 - x_next(N-1)) - dt*mu2/dx/dx*(0 - 2*x_next(N) + x_next(N-1));
        r(2:N-1) = x_next(2:N-1) - x_curr(2:N-1) + dt/(2*dx)*x_next(2:N-1).*(x_next(3:N) - x_next(1:N-2)) - dt*mu2/dx/dx*(x_next(1:N-2) - 2*x_next(2:N-1) + x_next(3:N));
        
        J(1,1) = 1 + dt/2/dx*(x_next(2)-mu_left) + 2*dt*mu2/dx/dx;
        J(1,2) = dt/2/dx*x_next(1) - mu2*dt/dx/dx;

        J(N,N) = 1 + dt/2/dx*(0-x_next(N-1)) + 2*dt*mu2/dx/dx;
        J(N,N-1) = -dt/2/dx*x_next(N) - mu2*dt/dx/dx;

        for i=2:N-1
            J(i,i) = 1 + dt/2/dx*(x_next(i+1)-x_next(i-1)) + 2*dt*mu2/dx/dx;
            J(i,i-1) = -dt/2/dx*x_next(i) - mu2*dt/dx/dx;
            J(i,i+1) = dt/2/dx*x_next(i) - mu2*dt/dx/dx;
        end

        Psi = J*phi;
        LHS = Psi'*Psi;
        RHS = Psi'*r;
            
            
        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
            iterCount = iterCount+1;
            if iterCount>50
                x_hat_hist(:,t:end) = nan*zeros(size(x_hat_hist(:,t:end)));
                break;
            end
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
    end
end
ROMtime = toc;

end



function [x_hat_hist, tTotal] = getSolutionHFGalerkinOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,C,A,F,B,M,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);

A = mu2*A;
B = mu2*B;


G = zeros(n,n*n,n);
H = zeros(n,n*n,n);

for i=1:n
    for j=1:n
        H(i,n*(j-1)+i,j) = 1;
    end
    G(i,(i-1)*n+1:i*n,:) = eye(n);
end

% precompute kron(phi,phi) for efficiency
F_phi_kron_phi = getF_phi_kron_phi(F,phi);


Gamma1 = phi'*A*phi;
Gamma2 = zeros(n,n,n);
Gamma3 = zeros(n,n,n);
Gamma4 = phi'*M*(kron(eye(1),phi));
Gamma5 = phi'*C;
Gamma6 = phi'*A*phi;
Gamma7 = phi'*F_phi_kron_phi;
Gamma8 = phi'*B;
Gamma9 = phi'*M*kron(eye(1),phi);
Gamma_2_3 = zeros(n,n,n);

for i=1:n
    Gamma2(i,:,:) = phi'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma3(i,:,:) = phi'*F_phi_kron_phi*squeeze(G(i,:,:));

    Gamma_2_3(i,:,:) = Gamma2(i,:,:) + Gamma3(i,:,:);
end

tic
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        u_ = mu_left;
        
        x_kron_x = reshape(x_hat_next * x_hat_next', [], 1);
        LHS = eye(n) - dt*(Gamma1 + Gamma4*u_ + tensorprod(Gamma_2_3,x_hat_next,1,1));
        RHS = x_hat_next - x_hat_curr - dt*(Gamma5 + Gamma6*x_hat_next + Gamma7*x_kron_x + Gamma8*u_ + Gamma9*u_*x_hat_next);

        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
    end
end
tTotal = toc;

end


function [x_hat_hist,totalTime] = getSolutionHFLSPGOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x,C,A,F,B,M,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);

A = mu2*A;
B = mu2*B;

G = zeros(n,n*n,n);
H = zeros(n,n*n,n);

for i=1:n
    for j=1:n
        H(i,n*(j-1)+i,j) = 1;
    end
    G(i,(i-1)*n+1:i*n,:) = eye(n);
end

% precompute kron(phi,phi) for efficiency
F_phi_kron_phi = getF_phi_kron_phi(F,phi);

% LHS
Gamma1 = phi'*A*phi;
Gamma2 = zeros(n,n,n);
Gamma3 = zeros(n,n,n);
Gamma4 = phi'*M*kron(eye(1),phi);
Gamma5 = phi'*A'*phi;
Gamma6 = phi'*A'*A*phi;
Gamma7 = zeros(n,n,n);
Gamma8 = zeros(n,n,n);
Gamma9 = phi'*A'*M*kron(eye(1),phi);
Gamma10 = zeros(n,n,n);
Gamma11 = zeros(n,n,n);
Gamma12 = zeros(n,n,n,n);
Gamma13 = zeros(n,n,n,n);
Gamma14 = zeros(n,n,1*n);
Gamma15 = zeros(n,n,n);
Gamma16 = zeros(n,n,n);
Gamma17 = zeros(n,n,n,n);
Gamma18 = zeros(n,n,n,n);
Gamma19 = zeros(n,n,1*n);
Gamma20 = (kron(eye(1),phi))'*M'*phi;
Gamma21 = (kron(eye(1),phi))'*M'*A*phi;
Gamma22 = zeros(n,1*n,n);
Gamma23 = zeros(n,1*n,n);
Gamma24 = (kron(eye(1),phi))'*M'*M*(kron(eye(1),phi));

% RHS
Gamma25 = phi'*C;
Gamma26 = phi'*A*phi;
Gamma27 = phi'*F_phi_kron_phi;
Gamma28 = phi'*B;
Gamma29 = phi'*M*kron(eye(1),phi);
Gamma30 = phi'*A'*phi;
Gamma31 = phi'*A'*C;
Gamma32 = phi'*A'*A*phi;
Gamma33 = phi'*A'*F_phi_kron_phi;
Gamma34 = phi'*A'*B;
Gamma35 = phi'*A'*M*kron(eye(1),phi);
Gamma36 = zeros(n,n,n);
Gamma37 = zeros(n,n,1);
Gamma38 = zeros(n,n,n);
Gamma39 = zeros(n,n,n*n);
Gamma40 = zeros(n,n,1);
Gamma41 = zeros(n,n,1*n);
Gamma42 = zeros(n,n,n);
Gamma43 = zeros(n,n,1);
Gamma44 = zeros(n,n,n);
Gamma45 = zeros(n,n,n*n);
Gamma46 = zeros(n,n,1);
Gamma47 = zeros(n,n,1*n);
Gamma48 = (kron(eye(1),phi))'*M'*phi;
Gamma49 = (kron(eye(1),phi))'*M'*C;
Gamma50 = (kron(eye(1),phi))'*M'*A*phi;
Gamma51 = (kron(eye(1),phi))'*M'*F_phi_kron_phi;
Gamma52 = (kron(eye(1),phi))'*M'*B;
Gamma53 = (kron(eye(1),phi))'*M'*M*kron(eye(1),phi);



for i=1:n
    % LHS
    Gamma2(i,:,:) = phi'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma3(i,:,:) = phi'*F_phi_kron_phi*squeeze(G(i,:,:));
    Gamma7(i,:,:) = phi'*A'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma8(i,:,:) = phi'*A'*F_phi_kron_phi*squeeze(G(i,:,:));
    Gamma10(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*phi;
    Gamma11(i,:,:) = squeeze(H(i,:,:))'*(F_phi_kron_phi)'*A*phi;
    Gamma14(i,:,:) = squeeze(H(i,:,:))'*(F_phi_kron_phi)'*M*(kron(eye(1),phi));
    Gamma15(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*phi;
    Gamma16(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*A*phi;
    Gamma19(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*M*(kron(eye(1),phi));
    Gamma22(i,:,:) = kron(eye(1),phi)'*M'*(F_phi_kron_phi)*squeeze(H(i,:,:));
    Gamma23(i,:,:) = kron(eye(1),phi)'*M'*(F_phi_kron_phi)*squeeze(G(i,:,:));

    for j=1:n
        Gamma12(i,j,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi*squeeze(H(j,:,:));
        Gamma13(i,j,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi*squeeze(G(j,:,:));
        Gamma17(i,j,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi*squeeze(H(j,:,:));
        Gamma18(i,j,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi*squeeze(G(j,:,:));
    end

    % RHS
    Gamma36(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*phi;
    Gamma37(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*C;
    Gamma38(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*A*phi;
    Gamma39(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi;
    Gamma40(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*B;
    Gamma41(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*M*kron(eye(1),phi);
    Gamma42(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*phi;
    Gamma43(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*C;
    Gamma44(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*A*phi;
    Gamma45(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi;
    Gamma46(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*B;
    Gamma47(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*M*kron(eye(1),phi);


end

tic;
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        u_ = mu_left;
        
        u_kron_I = kron(u_,eye(n));
        x_kron_x = reshape(x_hat_next * x_hat_next', [], 1);
        u_kron_x = kron(u_, x_hat_next);

        LHS = eye(n) - dt*Gamma1 - dt*Gamma5 + dt*dt*Gamma6;
        LHS = LHS + (-dt*Gamma4 + dt*dt*Gamma9)*u_;
        LHS = LHS + u_*(-dt*Gamma20 + dt*dt*Gamma21 + dt*dt*Gamma24*u_);
        
        LHS = LHS + tensorprod(-dt*Gamma2-dt*Gamma3+dt*dt*Gamma7+dt*dt*Gamma8-dt*Gamma10+dt*dt*Gamma11-dt*Gamma15+dt*dt*Gamma16,x_hat_next,1,1);
        LHS = LHS + tensorprod(dt*dt*Gamma14+dt*dt*Gamma19,x_hat_next,1,1)*u_;
        LHS = LHS + u_*tensorprod(dt*dt*Gamma22+dt*dt*Gamma23,x_hat_next,1,1);
        LHS = LHS + tensorprod(tensorprod(dt*dt*(Gamma12+Gamma13+Gamma17+Gamma18),x_hat_next,1,1),x_hat_next,1,1);
        
        
        RHS = x_hat_next - x_hat_curr;
        RHS = RHS + (-dt*Gamma25 + dt*dt*Gamma31);
        RHS = RHS + (-dt*Gamma26 - dt*Gamma30 + dt*dt*Gamma32)*x_hat_next + dt*Gamma30*x_hat_curr;
        RHS = RHS + (-dt*Gamma27 + dt*dt*Gamma33)*x_kron_x;
        RHS = RHS + (-dt*Gamma28 + dt*dt*Gamma34)*u_;
        RHS = RHS + (-dt*Gamma29 + dt*dt*Gamma35)*u_kron_x;

        RHS = RHS + u_*((-dt*Gamma48 + dt*dt*Gamma50)*x_hat_next + (dt*Gamma48*x_hat_curr) + (dt*dt*Gamma49));
        RHS = RHS + u_'*(dt*dt*Gamma51*x_kron_x + dt*dt*Gamma52*u_ + dt*dt*Gamma53*u_kron_x);
        
        RHS = RHS + tensorprod(-dt*Gamma36+dt*dt*Gamma38-dt*Gamma42+dt*dt*Gamma44,x_hat_next,1,1)*x_hat_next;
        RHS = RHS + tensorprod(dt*Gamma36+dt*Gamma42,x_hat_next,1,1)*x_hat_curr;
        RHS = RHS + tensorprod(dt*dt*Gamma37'+dt*dt*Gamma43',x_hat_next,2,1);

        RHS = RHS + tensorprod(dt*dt*Gamma39+dt*dt*Gamma45,x_hat_next,1,1)*x_kron_x;
        RHS = RHS + tensorprod(dt*dt*Gamma41+dt*dt*Gamma47,x_hat_next,1,1)*u_kron_x;
        RHS = RHS + tensorprod(dt*dt*Gamma40+dt*dt*Gamma46,x_hat_next,1,1)*u_;
        
        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
    end
end
totalTime = toc;

end


function [C, A, F, B, M] = getCoeffMat(N,x,dx)

C = zeros(N,1);
A = zeros(N,N);
F = zeros(N,N*N);
B = zeros(N,1);
M = zeros(N,1*N);


M(1,1) = 1/(2*dx); % mu_LHS term for left boundary in the advection term
B(1,1) = 1/(dx*dx); % mu_LHS term for the left boundary in the diffusion term


for i=1:N
    A(i,i) = -2/dx/dx;
    
    if i~=1
        A(i,i-1) = 1/dx/dx;
        F(i,(i-1)*N+(i-1)) = 1/(2*dx);
    end
    if i~=N
        A(i,i+1) = 1/dx/dx;
        F(i,(i-1)*N+(i+1)) = -1/(2*dx);
    end
    
end

A = sparse(A);
F = sparse(F);
B = sparse(B);
M = sparse(M);
C = sparse(C);

end


function [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x)

x = x';
numModes = size(phi,2);
x_hat_hist = zeros(numModes,Nt);

dx = x(2)-x(1);


n = size(phi,2);

indices_p = indices;
for i=1:length(indices)
    if ~ismember(indices(i)-1, indices_p) && indices(i)~=1
        indices_p = [indices_p; indices(i)-1];
    end
    if ~ismember(indices(i)+1, indices_p) && indices(i)~=N
        indices_p = [indices_p; indices(i)+1];
    end
end

neighbor_set = zeros(N,2);

for i=1:length(indices)
    if indices(i)~=1 && indices(i)~=N
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)-1
                neighbor_set(indices(i), 1) = j;
            elseif indices_p(j)==indices(i)+1
                neighbor_set(indices(i), 2) = j;
            end
        end
    elseif indices(i)==1
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)+1
                neighbor_set(indices(i), 2) = j;
            end
        end
    elseif indices(i)==N
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)-1
                neighbor_set(indices(i), 1) = j;
            end
        end
    end
end

indices_to_indices_p = zeros(length(indices),1);

for i=1:length(indices)
    for j=1:length(indices_p)
        if indices(i)==indices_p(j)
            indices_to_indices_p(i) = j;
            break
        end
    end
end

phi_sparse = phi(indices,:);
phi_sparse_p = phi(indices_p,:);

tic
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);
    
    iterCount = 0;
    while true
        x_curr_sparse = phi_sparse*x_hat_curr;
        x_next_sparse = phi_sparse*x_hat_next;

        x_next_sparse_p = phi_sparse_p*x_hat_next;

        r_sparse = zeros(size(indices,1), 1);
        J_sparse = zeros(size(indices,1), size(indices_p,1));


        for i=1:length(indices)
            
            if indices(i)==1
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(x_next_sparse_p(neighbor_set(indices(i),2)) - mu_left) - dt*mu2/dx/dx*(mu_left - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2)));
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(x_next_sparse_p(neighbor_set(indices(i),2))-mu_left) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;
                
            elseif indices(i)==N
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(0 - x_next_sparse_p(neighbor_set(indices(i),1))) - dt*mu2/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + 0);
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(0-x_next_sparse_p(neighbor_set(indices(i),1))) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),1)) = -dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;

            else
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(x_next_sparse_p(neighbor_set(indices(i),2)) - x_next_sparse_p(neighbor_set(indices(i),1))) - dt*mu2/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2)));
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(x_next_sparse_p(neighbor_set(indices(i),2))-x_next_sparse_p(neighbor_set(indices(i),1))) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),1)) = -dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;

            end
        end
        
        psi_sparse = J_sparse*phi_sparse_p;

        psi_weighted = xi .* psi_sparse;
        r_sparse_approx= tensorprod(psi_weighted, r_sparse, 1, 1);
        J_sparse_approx = tensorprod(psi_weighted, psi_sparse, 1, 1);

        
        LHS = J_sparse_approx;
        RHS = r_sparse_approx;

        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
        iterCount = iterCount + 1;
        if iterCount==50
            x_hat_hist = NaN*ones(size(x_hat_hist));
            ROMtime = NaN;
            return
        end
    end
end
ROMtime = toc;


end


function [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,phi,xi,indices,x)

x = x';
numModes = size(phi,2);
x_hat_hist = zeros(numModes,Nt);

dx = x(2)-x(1);


n = size(phi,2);

indices_p = indices;
for i=1:length(indices)
    if ~ismember(indices(i)-1, indices_p) && indices(i)~=1
        indices_p = [indices_p; indices(i)-1];
    end
    if ~ismember(indices(i)+1, indices_p) && indices(i)~=N
        indices_p = [indices_p; indices(i)+1];
    end
end

neighbor_set = zeros(N,2);

for i=1:length(indices)
    if indices(i)~=1 && indices(i)~=N
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)-1
                neighbor_set(indices(i), 1) = j;
            elseif indices_p(j)==indices(i)+1
                neighbor_set(indices(i), 2) = j;
            end
        end
    elseif indices(i)==1
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)+1
                neighbor_set(indices(i), 2) = j;
            end
        end
    elseif indices(i)==N
        for j=1:length(indices_p)
            if indices_p(j)==indices(i)-1
                neighbor_set(indices(i), 1) = j;
            end
        end
    end
end

indices_to_indices_p = zeros(length(indices),1);

for i=1:length(indices)
    for j=1:length(indices_p)
        if indices(i)==indices_p(j)
            indices_to_indices_p(i) = j;
            break
        end
    end
end

phi_sparse = phi(indices,:);
phi_sparse_p = phi(indices_p,:);

tic
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);
    
    iterCount = 0;
    while true
        x_curr_sparse = phi_sparse*x_hat_curr;
        x_next_sparse = phi_sparse*x_hat_next;

        x_next_sparse_p = phi_sparse_p*x_hat_next;

        r_sparse = zeros(size(indices,1), 1);
        J_sparse = zeros(size(indices,1), size(indices_p,1));

        for i=1:length(indices)
            
            if indices(i)==1
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(x_next_sparse_p(neighbor_set(indices(i),2)) - mu_left) - dt*mu2/dx/dx*(mu_left - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2)));
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(x_next_sparse_p(neighbor_set(indices(i),2))-mu_left) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;
                
            elseif indices(i)==N
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(0 - x_next_sparse_p(neighbor_set(indices(i),1))) - dt*mu2/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + 0);
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(0-x_next_sparse_p(neighbor_set(indices(i),1))) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),1)) = -dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;


            else
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) + dt/(2*dx)*x_next_sparse(i).*(x_next_sparse_p(neighbor_set(indices(i),2)) - x_next_sparse_p(neighbor_set(indices(i),1))) - dt*mu2/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2)));
                J_sparse(i,indices_to_indices_p(i)) = 1 + dt/2/dx*(x_next_sparse_p(neighbor_set(indices(i),2))-x_next_sparse_p(neighbor_set(indices(i),1))) + 2*dt*mu2/dx/dx;
                J_sparse(i,neighbor_set(indices(i),1)) = -dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = dt/2/dx*x_next_sparse(i) - mu2*dt/dx/dx;

            end
        end
        
        psi_sparse = J_sparse*phi_sparse_p;

        phi_weighted = xi .* phi_sparse;
        r_sparse_approx = tensorprod(phi_weighted, r_sparse, 1, 1);
        J_sparse_approx = tensorprod(phi_weighted, psi_sparse, 1, 1);

        LHS = J_sparse_approx;
        RHS = r_sparse_approx;
        
        if norm(RHS)>=tol
            x_hat_next = x_hat_next - LHS\RHS;
        else
            x_hat_hist(:,t) = x_hat_next;
            break
        end
        iterCount = iterCount + 1;
        if iterCount==50
            x_hat_hist = NaN*ones(size(x_hat_hist));
            ROMtime = NaN;
            return
        end
    end
end
ROMtime = toc;


end




function x_hist = getAllTrainingSolutionsOptimized(N,dt,Nt,tol,x,dx)

x_hist = [];

for i=1:10
    mu_left = 1 + .25*(i-1);
    for j=1:10
        mu2 = .01 + .01*(j-1);
        [x_hist_sol, FOMtime] = getSolutionOptimized(N,dt,Nt,tol,mu_left,mu2,dx,x);
        x_hist = [x_hist x_hist_sol];
    end
end

end




function [F_phi_kron_phi] = getF_phi_kron_phi(F,phi)
N = size(phi,1);
n = size(phi,2);

F_phi_kron_phi = zeros(N,n*n);
[ii,jj,ss] = find(F);


for k=1:size(ii,1)
    i=ii(k);
    j=jj(k);
    s=ss(k);

    iL = i;
    for iR=1:n*n
        iBlock = floor((j-1)/(N))+1;
        jBlock = floor((iR-1)/n)+1;
        iSubBlock = rem(j-1,N)+1;
        jSubBlock = rem(iR-1,n)+1;
        F_phi_kron_phi(iL,iR) = F_phi_kron_phi(iL,iR) + s*phi(iBlock,jBlock)*phi(iSubBlock,jSubBlock);
    end

end


end


