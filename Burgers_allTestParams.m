clc;
clear all;
close all;

% user inputs
N = 1024;
dt = .001;
Nt = 500;

N1 = 10; % number of mu1
N2 = 10; % number of mu2

Nh = 5000; % number of sample solutions used to get the ECSW weights (following the procedure of Grimberg, 2020)

dx = 1/(N+1);
x = linspace(dx,1-dx,N);
tol = 1e-6;

t_domain = linspace(0,dt*Nt,Nt+1);
[X, T] = meshgrid(x,t_domain);


[C, A, F, B, M] = getCoeffMat(N,x,dx);

x_hist = getAllTrainingSolutionsOptimized(N,dt,Nt+1,tol,x,dx,N1,N2);

[x_hist_test, FOMtime] = getAllTestSolutionsOptimized(N,dt,Nt+1,tol,x,dx,N1,N2);



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
statePredictionErrorHFLSPG = zeros(size(latTol,1),N1-1,N2-1);
statePredictionErrorHFGalerkin = zeros(size(latTol,1),N1-1,N2-1);
statePredictionErrorECSWLSPG1e5 = zeros(size(latTol,1),N1-1,N2-1);
statePredictionErrorECSWLSPG1e9 = zeros(size(latTol,1),N1-1,N2-1);
statePredictionErrorECSWGalerkin1e5 = zeros(size(latTol,1),N1-1,N2-1);
statePredictionErrorECSWGalerkin1e9 = zeros(size(latTol,1),N1-1,N2-1);



for i=1:size(latTol,1)
    i
    for j=1:size(uStan_error,1)
        if uStan_error(j)<latTol(i)
            numModes = j;
            break
        end
    end

    phi = U(:,1:numModes);

    [x_hat_HFGalerkin, ROMtime] = getSolutionHFGalerkinOptimized(N,dt,Nt+1,tol,dx,x,C,A,F,B,M,phi,N1,N2);
    x_approx_HFGalerkin = getApprox(phi,x_hat_HFGalerkin,N1-1,N2-1);
    statePredictionErrorHFGalerkin(i,:,:) = getError(x_approx_HFGalerkin,x_hist_test,N1-1,N2-1);


    [x_hat_HFLSPG, ROMtime] = getSolutionHFLSPGOptimized(N,dt,Nt+1,tol,dx,x,C,A,F,B,M,phi,N1,N2);
    x_approx_HFLSPG = getApprox(phi,x_hat_HFLSPG,N1-1,N2-1);
    statePredictionErrorHFLSPG(i,:,:) = getError(x_approx_HFLSPG,x_hist_test,N1-1,N2-1);


    load(strcat('BurgersECSWweights/LSPG_1e5_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt+1,tol,phi,xi,indices,x,N1,N2);
    x_approx_ECSWLSPG1e5 = getApprox(phi,x_hat_hist,N1-1,N2-1);
    statePredictionErrorECSWLSPG1e5(i,:,:) = getError(x_approx_ECSWLSPG1e5,x_hist_test,N1-1,N2-1);


    load(strcat('BurgersECSWweights/Galerkin_1e5_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt+1,tol,phi,xi,indices,x,N1,N2);
    x_approx_ECSWGalerkin1e5 = getApprox(phi,x_hat_hist,N1-1,N2-1);
    statePredictionErrorECSWGalerkin1e5(i,:,:) = getError(x_approx_ECSWGalerkin1e5,x_hist_test,N1-1,N2-1);


    load(strcat('BurgersECSWweights/LSPG_1e9_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt+1,tol,phi,xi,indices,x,N1,N2);
    x_approx_ECSWLSPG1e9 = getApprox(phi,x_hat_hist,N1-1,N2-1);
    statePredictionErrorECSWLSPG1e9(i,:,:) = getError(x_approx_ECSWLSPG1e9,x_hist_test,N1-1,N2-1);


    load(strcat('BurgersECSWweights/Galerkin_1e9_', int2str(i), '.mat'),'xi','indices');
    [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt+1,tol,phi,xi,indices,x,N1,N2);
    x_approx_ECSWGalerkin1e9 = getApprox(phi,x_hat_hist,N1-1,N2-1);
    statePredictionErrorECSWGalerkin1e9(i,:,:) = getError(x_approx_ECSWGalerkin1e9,x_hist_test,N1-1,N2-1);
  
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



function [x_hat_hist, tTotal] = getSolutionHFGalerkinOptimized(N,dt,Nt,tol,dx,x,C,A,F,B,M,phi,N1,N2)

n = size(phi,2);
x_hat_hist = zeros(N1-1,N2-1,n,Nt);
tTotal = zeros(N1-1,N2-1);

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

for i=1:n
    Gamma2(i,:,:) = phi'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma3(i,:,:) = phi'*F_phi_kron_phi*squeeze(G(i,:,:));
end

for i=1:N1-1
    mu1 = 1.125 + .25*(i-1);
    for j=1:N2-1
        mu2 = .015  + .01*(j-1);
        
        %%%% time series loop
        tic
        Gamma1_mu2 = Gamma1*mu2;
        Gamma6_mu2 = Gamma6*mu2;
        Gamma8_mu2 = Gamma8*mu2;
        

        for t=2:Nt
            x_hat_curr = squeeze(x_hat_hist(i,j,:,t-1));
            x_hat_next = squeeze(x_hat_hist(i,j,:,t-1));
        
            while true
                u_ = mu1;
        
                LHS = eye(n) - dt*(Gamma1_mu2 + Gamma4*kron(u_,eye(n)) + tensorprod(Gamma2,x_hat_next,1,1) + tensorprod(Gamma3,x_hat_next,1,1));
                RHS = x_hat_next - x_hat_curr - dt*(Gamma5 + Gamma6_mu2*x_hat_next + Gamma7*(kron(x_hat_next, x_hat_next)) + Gamma8_mu2*u_ + Gamma9*(kron(u_, x_hat_next)));
                
                if norm(RHS)>=tol
                    x_hat_next = x_hat_next - LHS\RHS;
                else
                    x_hat_hist(i,j,:,t) = x_hat_next;
                    break
                end
            end
        end
        tTotal(i,j) = toc;


    end
end



end




function [x_hat_hist,totalTime] = getSolutionHFLSPGOptimized(N,dt,Nt,tol,dx,x,C,A,F,B,M,phi,N1,N2)

n = size(phi,2);

x_hat_hist = zeros(N1-1,N2-1,n,Nt);
totalTime = zeros(N1-1,N2-1);

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

for i=1:N1-1
    mu1 = 1.125 + .25*(i-1);
    for j=1:N2-1
        mu2 = .015  + .01*(j-1);
        
        tic
        Gamma1_mu2 = Gamma1*mu2;
        Gamma5_mu2 = Gamma5*mu2;
        Gamma9_mu2 = Gamma9*mu2;
        Gamma21_mu2 = Gamma21*mu2;
        Gamma26_mu2 = Gamma26*mu2;
        Gamma28_mu2 = Gamma28*mu2;
        Gamma30_mu2 = Gamma30*mu2;
        Gamma31_mu2 = Gamma31*mu2;
        Gamma33_mu2 = Gamma33*mu2;
        Gamma35_mu2 = Gamma35*mu2;
        Gamma50_mu2 = Gamma50*mu2;
        Gamma52_mu2 = Gamma52*mu2;
        Gamma7_mu2 = Gamma7*mu2;
        Gamma8_mu2 = Gamma8*mu2;
        Gamma11_mu2 = Gamma11*mu2;
        Gamma16_mu2 = Gamma16*mu2;
        Gamma38_mu2 = Gamma38*mu2;
        Gamma40_mu2 = Gamma40*mu2;
        Gamma44_mu2 = Gamma44*mu2;
        Gamma46_mu2 = Gamma46*mu2;

        Gamma6_mu2_mu2 = Gamma6*mu2*mu2;
        Gamma32_mu2_mu2 = Gamma32*mu2*mu2;
        Gamma34_mu2_mu2 = Gamma34*mu2*mu2;

        




        for t=2:Nt
            x_hat_curr = squeeze(x_hat_hist(i,j,:,t-1));
            x_hat_next = squeeze(x_hat_hist(i,j,:,t-1));
        
            while true
                u_ = mu1;
                
                u_kron_I = kron(u_,eye(n));
                x_kron_x = kron(x_hat_next, x_hat_next);
                u_kron_x = kron(u_, x_hat_next);
        
                LHS = eye(n) - dt*Gamma1_mu2 - dt*Gamma5_mu2 + dt*dt*Gamma6_mu2_mu2;
                LHS = LHS + (-dt*Gamma4 + dt*dt*Gamma9_mu2)*u_kron_I;
                LHS = LHS + u_kron_I'*(-dt*Gamma20 + dt*dt*Gamma21_mu2 + dt*dt*Gamma24*u_kron_I);
                
                LHS = LHS + tensorprod(-dt*Gamma2-dt*Gamma3+dt*dt*Gamma7_mu2+dt*dt*Gamma8_mu2-dt*Gamma10+dt*dt*Gamma11_mu2-dt*Gamma15+dt*dt*Gamma16_mu2,x_hat_next,1,1);
                LHS = LHS + tensorprod(dt*dt*Gamma14+dt*dt*Gamma19,x_hat_next,1,1)*u_kron_I;
                LHS = LHS + tensorprod(permute(pagemtimes(u_kron_I',dt*dt*permute(Gamma22,[2,3,1])+dt*dt*permute(Gamma23,[2,3,1])),[3,1,2]),x_hat_next,1,1);
                LHS = LHS + tensorprod(tensorprod(dt*dt*(Gamma12+Gamma13+Gamma17+Gamma18),x_hat_next,1,1),x_hat_next,1,1);
                
                
                RHS = x_hat_next - x_hat_curr;
                RHS = RHS + (-dt*Gamma25 + dt*dt*Gamma31_mu2);
                RHS = RHS + (-dt*Gamma26_mu2 - dt*Gamma30_mu2 + dt*dt*Gamma32_mu2_mu2)*x_hat_next + dt*Gamma30_mu2*x_hat_curr;
                RHS = RHS + (-dt*Gamma27 + dt*dt*Gamma33_mu2)*x_kron_x;
                RHS = RHS + (-dt*Gamma28_mu2 + dt*dt*Gamma34_mu2_mu2)*u_;
                RHS = RHS + (-dt*Gamma29 + dt*dt*Gamma35_mu2)*u_kron_x;
        
                RHS = RHS + u_kron_I'*((-dt*Gamma48 + dt*dt*Gamma50_mu2)*x_hat_next + (dt*Gamma48*x_hat_curr) + (dt*dt*Gamma49));
                RHS = RHS + u_kron_I'*(dt*dt*Gamma51*x_kron_x + dt*dt*Gamma52_mu2*u_ + dt*dt*Gamma53*u_kron_x);
                
                RHS = RHS + tensorprod(-dt*Gamma36+dt*dt*Gamma38_mu2-dt*Gamma42+dt*dt*Gamma44_mu2,x_hat_next,1,1)*x_hat_next;
                RHS = RHS + tensorprod(dt*Gamma36+dt*Gamma42,x_hat_next,1,1)*x_hat_curr;
                RHS = RHS + tensorprod(dt*dt*Gamma37'+dt*dt*Gamma43',x_hat_next,2,1);
        
                RHS = RHS + tensorprod(dt*dt*Gamma39+dt*dt*Gamma45,x_hat_next,1,1)*x_kron_x;
                RHS = RHS + tensorprod(dt*dt*Gamma41+dt*dt*Gamma47,x_hat_next,1,1)*u_kron_x;
                RHS = RHS + tensorprod(dt*dt*Gamma40_mu2+dt*dt*Gamma46_mu2,x_hat_next,1,1)*u_;
        
                if norm(RHS)>=tol
                    x_hat_next = x_hat_next - LHS\RHS;
                else
                    x_hat_hist(i,j,:,t) = x_hat_next;
                    break
                end
            end
        end
        totalTime(i,j) = toc;
    end
end

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



function [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,phi,xi,indices,x,N1,N2)

x = x';
numModes = size(phi,2);

x_hat_hist = zeros(N1-1,N2-1,numModes,Nt);
ROMtime = zeros(N1-1,N2-1);

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


for j=1:N1-1
    mu_left = 1.125 + .25*(j-1);
    for k=1:N2-1
        mu2 = .015  + .01*(k-1);

        flag=0;
        tic
        for t=2:Nt
            x_hat_curr = squeeze(x_hat_hist(j,k,:,t-1));
            x_hat_next = squeeze(x_hat_hist(j,k,:,t-1));
            
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
                    x_hat_hist(j,k,:,t) = x_hat_next;
                    break
                end
                iterCount = iterCount + 1;
                if iterCount==50
                    x_hat_hist(j,k,:,:) = NaN*ones(numModes,Nt);
                    ROMtime(j,k) = NaN;
                    flag=1;
                    break
                end
            end
            if flag==1
                break
            end
        end
        ROMtime(j,k) = toc;
    end
end


end


function [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,phi,xi,indices,x,N1,N2)

x = x';
numModes = size(phi,2);

x_hat_hist = zeros(N1-1,N2-1,numModes,Nt);
ROMtime = zeros(N1-1,N2-1);

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

for j=1:N1-1
    mu_left = 1.125 + .25*(j-1);
    for k=1:N2-1
        mu2 = .015  + .01*(k-1);

        flag=0;

        tic
        
        for t=2:Nt
            x_hat_curr = squeeze(x_hat_hist(j,k,:,t-1));
            x_hat_next = squeeze(x_hat_hist(j,k,:,t-1));
            
            iterCount = 0;
            while true
                x_curr_sparse = phi_sparse*x_hat_curr;
                x_next_sparse = phi_sparse*x_hat_next;
        
                x_next_sparse_p = phi_sparse_p*x_hat_next;
        
                r_sparse = zeros(size(indices,1), 1);
                J_sparse = zeros(size(indices,1), size(indices_p,1));
        
                for i=1:length(indices)
                    
                    % r_sparse(1)
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
                
                % t
                % fprintf('%3.6f / %3.6f \n', norm(RHS), tol);
                if norm(RHS)>=tol
                    x_hat_next = x_hat_next - LHS\RHS;
                else
                    x_hat_hist(j,k,:,t) = x_hat_next;
                    break
                end
                iterCount = iterCount + 1;
                if iterCount==50
                    x_hat_hist(j,k,:,:) = NaN*ones(numModes,Nt);
                    ROMtime(j,k) = NaN;
                    flag = 1;
                    break
                end
            end
            if flag==1
                break
            end
        end
        ROMtime(j,k) = toc;
    end
end


end


function x_hist = getAllTrainingSolutionsOptimized(N,dt,Nt,tol,x,dx,N1,N2)

x_hist = [];

for i=1:N1
    mu1 = 1 + .25*(i-1);
    for j=1:N2
        mu2 = .01  + .01*(j-1);
        [x_hist_sol, FOMtime] = getSolutionOptimized(N,dt,Nt,tol,mu1,mu2,dx,x);
        x_hist = [x_hist x_hist_sol];
    end
end

end


function [x_hist_test, allTimes] = getAllTestSolutionsOptimized(N,dt,Nt,tol,x,dx,N1,N2)

x_hist_test = zeros(N1-1,N2-1,N,Nt);
allTimes = zeros(N1-1,N2-1);


for i=1:N1-1
    mu1 = 1.125 + .25*(i-1);
    for j=1:N2-1
        mu2 = .015  + .01*(j-1);
        [x_hist_sol, FOMtime] = getSolutionOptimized(N,dt,Nt,tol,mu1,mu2,dx,x);
        x_hist_test(i,j,:,:) = x_hist_sol;
        allTimes(i,j) = FOMtime;
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

function error = getError(x_approx,x_GT,N1,N2)

error = zeros(N1,N2);

for i=1:N1
    for j=1:N2
        error(i,j) = norm(squeeze(x_approx(i,j,:,:)) - squeeze(x_GT(i,j,:,:)),'fro')^2/norm(squeeze(x_GT(i,j,:,:)),'fro')^2;
    end
end

end


function approx = getApprox(phi,x_hat,N1,N2)

N = size(phi,1);
Nt = size(x_hat,4);

approx = zeros(N1-1,N2-1,N,Nt);
for i=1:N1
    for j=1:N2
        approx(i,j,:,:) = phi*squeeze(x_hat(i,j,:,:));
    end
end

end
