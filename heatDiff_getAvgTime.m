clc;
clear all;
close all;

% user inputs
N = 1024;
dt = .001;
Nt = 2000;
mu = .005;
a = 1.5;
b = .5;
Nh = 5000; % number of sample solutions used to get the ECSW weights (following the procedure of Grimberg, 2020)

nRep = 10;

dx = 1/(N+1);
x = linspace(dx,1-dx,N);
tol = 1e-6;
a_all = [-2 -1 0 1 2];
b_all = [0 -2 1 -1 2];
t_domain = linspace(0,dt*Nt,Nt);
[X, T] = meshgrid(x,t_domain);


[C, A, F, B, M] = getCoeffMat(N,x,dx,mu);
[Cn, An, Fn, Fcn, Bn, Mn] = getCoeffMatNoLift(N,x,dx,mu);

x_hist_sol1 = getSolutionOptimizedNoLift(N,dt,Nt,tol,-2,0,mu,dx,x);
x_hist_sol2 = getSolutionOptimizedNoLift(N,dt,Nt,tol,-1,-2,mu,dx,x);
x_hist_sol3 = getSolutionOptimizedNoLift(N,dt,Nt,tol,0,1,mu,dx,x);
x_hist_sol4 = getSolutionOptimizedNoLift(N,dt,Nt,tol,1,-1,mu,dx,x);
x_hist_sol5 = getSolutionOptimizedNoLift(N,dt,Nt,tol,2,2,mu,dx,x);
x_hist = [x_hist_sol1 x_hist_sol2 x_hist_sol3 x_hist_sol4 x_hist_sol5];


FOMtime = 0;
for i=1:nRep
    [x_hist_test, FOMtime_sol] = getSolutionOptimizedNoLift(N,dt,5*Nt,tol,a,b,mu,dx,x);
    FOMtime = FOMtime + FOMtime_sol;
end


%% lifted ROMs

[Ustan,SigmaStan,V] = svd(x_hist,'econ');
[U,Sigma,V] = svd(x_hist,'econ');
[Usq,SigmaSq,V] = svd(x_hist.^2,'econ');

latTol = [1e-2 1e-3 1e-4 1e-5 1e-6]';
u_error = zeros(N,1);
uSq_error = zeros(N,1);
uMon_error = zeros(2*N,1);
uStan_error = zeros(N,1);

for i=1:N
    u_error(i) = 1-sum(diag(Sigma(1:i,1:i)).^2) / sum(diag(Sigma).^2);
    uSq_error(i) = 1-sum(diag(SigmaSq(1:i,1:i)).^2) / sum(diag(SigmaSq).^2);
    uStan_error(i) = 1-sum(diag(SigmaStan(1:i,1:i)).^2) / sum(diag(SigmaStan).^2);
end

statePredictionErrorStandardLSPGlifted = zeros(size(latTol,1),1);
statePredictionErrorStandardGalerkinLifted = zeros(size(latTol,1),1);
statePredictionErrorHFGalerkinCubic = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e5 = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e7 = zeros(size(latTol,1),1);
statePredictionErrorECSWLSPG1e9 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e5 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e7 = zeros(size(latTol,1),1);
statePredictionErrorECSWGalerkin1e9 = zeros(size(latTol,1),1);


speedupFactorStandardGalerkin = zeros(size(latTol,1),1);
speedupFactorStandardLSPG = zeros(size(latTol,1),1);
speedupFactorHFGalerkinCubic = zeros(size(latTol,1),1);
speedupFactorHFLSPG = zeros(size(latTol,1),1);
speedupFactorHFGalerkin = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e5 = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e7 = zeros(size(latTol,1),1);
speedupFactorECSWLSPG1e9 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e5 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e7 = zeros(size(latTol,1),1);
speedupFactorECSWGalerkin1e9 = zeros(size(latTol,1),1);

dim_u = zeros(size(latTol,1),1);
dim_uSq = zeros(size(latTol,1),1);
dim_total = zeros(size(latTol,1),1);


for i=1:size(latTol,1)
    i
    for j=1:size(u_error,1)
        if u_error(j)<latTol(i)
            numModes_u = j;
            break
        end
    end
    for j=1:size(uSq_error,1)
        if uSq_error(j)<latTol(i)
            numModes_uSq = j;
            break
        end
    end
    for j=1:size(uMon_error,1)
        if uMon_error(j)<latTol(i)
            numModes_uMon = j;
            break
        end
    end

    for j=1:size(uStan_error,1)
        if uStan_error(j)<latTol(i)
            numModes_uStan = j;
            break
        end
    end

    dim_u(i) = numModes_u;
    dim_uSq(i) = numModes_uSq;
    dim_total(i) = numModes_u + numModes_uSq;

    phi = [U(:,1:numModes_u) zeros(N,numModes_uSq); zeros(N,numModes_u) Usq(:,1:numModes_uSq)];
    phiStan = Ustan(:,1:numModes_uStan);
    numSol = 5;


    for k=1:nRep
        x_hist_recon = phiStan*phiStan'*x_hist_test;
    
        [x_hat_Galerkin, ROMtime] = getStandardGalerkinSolutionOptimizedNoLift(N,dt,5*Nt,tol,a,b,mu,dx,x,phiStan);
        speedupFactorStandardGalerkin(i) = speedupFactorStandardGalerkin(i) + ROMtime;
    
    
        [x_hat_LSPG, ROMtime] = getStandardLSPGsolutionOptimizedNoLift(N,dt,5*Nt,tol,a,b,mu,dx,x,phiStan);
        speedupFactorStandardLSPG(i) = speedupFactorStandardLSPG(i) + ROMtime;
    
    
        [x_hat_HFGalerkinCubic, ROMtime] = getSolutionHFGalerkinCubicOptimized(N,dt,5*Nt,tol,a,b,mu,dx,x,Cn,An,Fcn,Bn,Mn,phiStan);
        speedupFactorHFGalerkinCubic(i) = speedupFactorHFGalerkinCubic(i) + ROMtime;
    
    
        [x_hat_HFGalerkin, ROMtime] = getSolutionLiftedHFGalerkinOptimized(N,dt,5*Nt,tol,a,b,mu,dx,x,C,A,F,B,M,phi);
        speedupFactorHFGalerkin(i) = speedupFactorHFGalerkin(i) + ROMtime;
    
        [x_hat_HFLSPG, ROMtime] = getSolutionLiftedHFLSPGOptimized(N,dt,5*Nt,tol,a,b,mu,dx,x,C,A,F,B,M,phi);
        speedupFactorHFLSPG(i) = speedupFactorHFLSPG(i) + ROMtime;
    
        load(strcat('HeatDiffECSWweights/LSPG_1e5_', int2str(i), '.mat'),'xi','indices');
        [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,5*Nt,tol,mu,a,b,phiStan,xi,indices,x);
        speedupFactorECSWLSPG1e5(i) = speedupFactorECSWLSPG1e5(i) + ROMtime;
    

        load(strcat('HeatDiffECSWweights/Galerkin_1e5_', int2str(i), '.mat'),'xi','indices');
        [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,5*Nt,tol,mu,a,b,phiStan,xi,indices,x);
        speedupFactorECSWGalerkin1e5(i) = speedupFactorECSWGalerkin1e5(i) + ROMtime;
    

        load(strcat('HeatDiffECSWweights/LSPG_1e9_', int2str(i), '.mat'),'xi','indices');
        [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,5*Nt,tol,mu,a,b,phiStan,xi,indices,x);
        speedupFactorECSWLSPG1e9(i) = speedupFactorECSWLSPG1e9(i) + ROMtime;
    

        load(strcat('HeatDiffECSWweights/Galerkin_1e9_', int2str(i), '.mat'),'xi','indices');
        [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,5*Nt,tol,mu,a,b,phiStan,xi,indices,x);
        speedupFactorECSWGalerkin1e9(i) = speedupFactorECSWGalerkin1e9(i) + ROMtime;
    end
    
    
end


speedupFactorStandardGalerkin = FOMtime./speedupFactorStandardGalerkin;
speedupFactorStandardLSPG = FOMtime./speedupFactorStandardLSPG;
speedupFactorHFGalerkin = FOMtime./speedupFactorHFGalerkin;
speedupFactorHFGalerkinCubic = FOMtime./speedupFactorHFGalerkinCubic;
speedupFactorHFLSPG = FOMtime./speedupFactorHFLSPG;
speedupFactorECSWGalerkin1e5 = FOMtime./speedupFactorECSWGalerkin1e5;
speedupFactorECSWGalerkin1e9 = FOMtime./speedupFactorECSWGalerkin1e9;
speedupFactorECSWLSPG1e5 = FOMtime./speedupFactorECSWLSPG1e5;
speedupFactorECSWLSPG1e9 = FOMtime./speedupFactorECSWLSPG1e9;
avgFOMtime = FOMtime/nRep;

save('heatDiffSpeedups.mat', 'avgFOMtime', 'speedupFactorStandardGalerkin', 'speedupFactorHFGalerkinCubic', 'speedupFactorStandardLSPG', 'speedupFactorHFGalerkin', 'speedupFactorHFLSPG', 'speedupFactorECSWGalerkin1e5', 'speedupFactorECSWGalerkin1e9', 'speedupFactorECSWLSPG1e5', 'speedupFactorECSWLSPG1e9')



function [x_hist_sol,FOMtime] = getSolutionOptimizedNoLift(N,dt,Nt,tol,a,b,mu,dx,x)

x_hist_sol = zeros(N,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hist_sol(1:N,1) = init_q;

tic
for t=2:Nt
    x_curr = x_hist_sol(:,t-1);
    x_next = x_hist_sol(:,t-1);

    while true
        r = zeros(N,1);
        J = zeros(N,N);

        u = (a*sin(2*pi*(t-1)*dt) ./ (1 + 100*(x-1/4).^2) + b*sin(4*pi*(t-1)*dt) ./ (1 + 100*(x-3/4).^2))';
        r(1) = x_next(1) - x_curr(1) - dt*mu/dx/dx*(0 - 2*x_next(1) + x_next(2)) + dt*x_next(1).^3 - dt*u(1);
        r(end) = x_next(end) - x_curr(end) - dt*mu/dx/dx*(x_next(end-1) - 2*x_next(end) + 1) + dt*x_next(end).^3 - dt*u(end);
        r(2:end-1) = x_next(2:end-1) - x_curr(2:end-1) - dt*mu/dx/dx*(x_next(1:end-2) - 2*x_next(2:end-1) + x_next(3:end)) + dt*x_next(2:end-1).^3 - dt*u(2:end-1);
        
        J(1,1) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(1)^2;
        J(1,2) = - dt*mu/dx/dx;
        
        J(end,end) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(end)^2;
        J(end,end-1) = - dt*mu/dx/dx;


        for i=2:N-1
            J(i,i) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next(i)*x_next(i);
            J(i,i-1) = -dt*mu/dx/dx;
            J(i,i+1) = -dt*mu/dx/dx;
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


function [x_hat_hist,ROMtime] = getStandardGalerkinSolutionOptimizedNoLift(N,dt,Nt,tol,a,b,mu,dx,x,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hat_hist(:,1) = phi'*init_q';

tic;
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        x_curr = phi*x_hat_curr;
        x_next = phi*x_hat_next;
        r = zeros(N,1);
        J = zeros(N,N);

        u = (a*sin(2*pi*(t-1)*dt) ./ (1 + 100*(x-1/4).^2) + b*sin(4*pi*(t-1)*dt) ./ (1 + 100*(x-3/4).^2))';
        r(1) = x_next(1) - x_curr(1) - dt*mu/dx/dx*(0 - 2*x_next(1) + x_next(2)) + dt*x_next(1).^3 - dt*u(1);
        r(end) = x_next(end) - x_curr(end) - dt*mu/dx/dx*(x_next(end-1) - 2*x_next(end) + 1) + dt*x_next(end).^3 - dt*u(end);
        r(2:end-1) = x_next(2:end-1) - x_curr(2:end-1) - dt*mu/dx/dx*(x_next(1:end-2) - 2*x_next(2:end-1) + x_next(3:end)) + dt*x_next(2:end-1).^3 - dt*u(2:end-1);
        
        J(1,1) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(1)^2;
        J(1,2) = - dt*mu/dx/dx;
        
        J(end,end) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(end)^2;
        J(end,end-1) = - dt*mu/dx/dx;


        for i=2:N-1
            J(i,i) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next(i)*x_next(i);
            J(i,i-1) = -dt*mu/dx/dx;
            J(i,i+1) = -dt*mu/dx/dx;
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



function [x_hat_hist,ROMtime] = getStandardLSPGsolutionOptimizedNoLift(N,dt,Nt,tol,a,b,mu,dx,x,phi)

n = size(phi,2);
x_hat_hist = zeros(n,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hat_hist(:,1) = phi'*init_q';

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

        u = (a*sin(2*pi*(t-1)*dt) ./ (1 + 100*(x-1/4).^2) + b*sin(4*pi*(t-1)*dt) ./ (1 + 100*(x-3/4).^2))';
        r(1) = x_next(1) - x_curr(1) - dt*mu/dx/dx*(0 - 2*x_next(1) + x_next(2)) + dt*x_next(1).^3 - dt*u(1);
        r(end) = x_next(end) - x_curr(end) - dt*mu/dx/dx*(x_next(end-1) - 2*x_next(end) + 1) + dt*x_next(end).^3 - dt*u(end);
        r(2:end-1) = x_next(2:end-1) - x_curr(2:end-1) - dt*mu/dx/dx*(x_next(1:end-2) - 2*x_next(2:end-1) + x_next(3:end)) + dt*x_next(2:end-1).^3 - dt*u(2:end-1);
        
        J(1,1) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(1)^2;
        J(1,2) = - dt*mu/dx/dx;
        
        J(end,end) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(end)^2;
        J(end,end-1) = - dt*mu/dx/dx;


        for i=2:N-1
            J(i,i) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next(i)*x_next(i);
            J(i,i-1) = -dt*mu/dx/dx;
            J(i,i+1) = -dt*mu/dx/dx;
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



function [x_hat_hist, tTotal] = getSolutionLiftedHFGalerkinOptimized(N,dt,Nt,tol,a,b,mu,dx,x,C,A,F,B,M,phi)

x_hist_sol = zeros(2*N,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x; %from mcquarrie 2025
init_w = init_q.^2;
x_hist_sol(1:N,1) = init_q;
x_hist_sol(N+1:2*N,1) = init_w;

n = size(phi,2);
x_hat_hist = zeros(n,Nt);
x_hat_hist(:,1) = phi'*x_hist_sol(:,1);

G = zeros(n,n*n,n);
H = zeros(n,n*n,n);

for i=1:n
    for j=1:n
        H(i,n*(j-1)+i,j) = 1;
    end
    G(i,(i-1)*n+1:i*n,:) = eye(n);
end

F_phi_kron_phi = getF_phi_kron_phi(F,phi);


Gamma1 = phi'*A*phi;
Gamma2 = zeros(n,n,n);
Gamma3 = zeros(n,n,n);
Gamma4 = phi'*M*(kron(eye(2),phi));
Gamma5 = phi'*C;
Gamma6 = phi'*A*phi;
Gamma7 = phi'*F_phi_kron_phi;
Gamma8 = phi'*B;
Gamma9 = phi'*M*kron(eye(2),phi);
Gamma_2_3 = zeros(n,n,n);

for i=1:n
    Gamma2(i,:,:) = phi'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma3(i,:,:) = phi'*F_phi_kron_phi*squeeze(G(i,:,:));
    Gamma_2_3(i,:,:) = Gamma2(i,:,:) + Gamma3(i,:,:);
end

u_kron_x = zeros(2*n,1);

tic
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        u_ = [a*sin(2*pi*(t-1)*dt); b*sin(4*pi*(t-1)*dt)];
        x_kron_x = reshape(x_hat_next * x_hat_next', [], 1);
        u_kron_x(1:n) = u_(1)*x_hat_next;
        u_kron_x(n+1:end) = u_(2)*x_hat_next;
        

        LHS = eye(n) - dt*(Gamma1 + Gamma4(:,1:n)*u_(1) + Gamma4(:,n+1:end)*u_(2) + tensorprod(Gamma_2_3,x_hat_next,1,1));
        RHS = x_hat_next - x_hat_curr - dt*(Gamma5 + Gamma6*x_hat_next + Gamma7*x_kron_x + Gamma8*u_ + Gamma9*u_kron_x);
        

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



function Ghat = get_phip_F_phi_kron_phi_kron_phi(phi)
N = size(phi,1);
n = size(phi,2);
RHS = zeros(N,n*n*n);

for i=1:N
    RHS(i,:) = kron(phi(i,:),kron(phi(i,:),phi(i,:)));
end

Ghat = -phi'*RHS;

end

function [x_hat_hist, tTotal] = getSolutionHFGalerkinCubicOptimized(N,dt,Nt,tol,a,b,mu,dx,x,C,A,Fc,B,M,phi)

x_hist_sol = zeros(N,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hist_sol(1:N,1) = init_q;

n = size(phi,2);
x_hat_hist = zeros(n,Nt);
x_hat_hist(:,1) = phi'*x_hist_sol(:,1);

G = zeros(n,n*n,n);
H = zeros(n,n*n,n);

for i=1:n
    for j=1:n
        H(i,n*(j-1)+i,j) = 1;
    end
    G(i,(i-1)*n+1:i*n,:) = eye(n);
end

Hc1 = zeros(n*n,n*n*n,n);
Hc2 = zeros(n*n,n*n*n,n);
Hc3 = zeros(n*n,n*n*n,n);

counter1 = 0;
counter2 = 1;
for i=1:n*n
    for j=1:n
        Hc1(i,(j-1)*n*n+i,j) = 1;
        Hc2(i,counter1*n*n+n*(j-1)+counter2,j) = 1;
        Hc3(i,(i-1)*n+j,j) = 1;
    end
    counter2 = counter2+1;
    if rem(i,n)==0
        counter1 = counter1+1;
        counter2 = 1;
    end
end

phip_F_phi_kron_phi_kron_phi = get_phip_F_phi_kron_phi_kron_phi(phi);

Gamma1 = phi'*A*phi;
Gamma2 = zeros(n,n,n);
Gamma3 = zeros(n,n,n);
Gamma4 = phi'*M*(kron(eye(2),phi));
Gamma5 = phi'*C;
Gamma6 = phi'*A*phi;
Gamma8 = phi'*B;
Gamma9 = phi'*M*kron(eye(2),phi);

Gamma10 = phip_F_phi_kron_phi_kron_phi;
Gamma11 = zeros(n*n,n,n);
Gamma12 = zeros(n*n,n,n);
Gamma13 = zeros(n*n,n,n);

Gamma_11_12_13 = zeros(n*n,n,n);

for i=1:n*n
    Gamma11(i,:,:) = phip_F_phi_kron_phi_kron_phi*squeeze(Hc1(i,:,:));
    Gamma12(i,:,:) = phip_F_phi_kron_phi_kron_phi*squeeze(Hc2(i,:,:));
    Gamma13(i,:,:) = phip_F_phi_kron_phi_kron_phi*squeeze(Hc3(i,:,:));

    Gamma_11_12_13(i,:,:) = Gamma11(i,:,:) + Gamma12(i,:,:) + Gamma13(i,:,:);
end

u_kron_x = zeros(2*n,1);

tic
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        u_ = [a*sin(2*pi*(t-1)*dt); b*sin(4*pi*(t-1)*dt)];
        x_kron_x = reshape(x_hat_next * x_hat_next', [], 1);
        x_kron_x_kron_x = reshape(x_hat_next * x_kron_x', [], 1);
        u_kron_x(1:n) = u_(1)*x_hat_next;
        u_kron_x(n+1:end) = u_(2)*x_hat_next;


        LHS = eye(n) - dt*(Gamma1 + Gamma4(:,1:n)*u_(1) + Gamma4(:,n+1:end)*u_(2) + tensorprod(Gamma_11_12_13,x_kron_x,1,1));
        RHS = x_hat_next - x_hat_curr - dt*(Gamma5 + Gamma6*x_hat_next + Gamma8*u_ + Gamma9*u_kron_x + Gamma10*x_kron_x_kron_x);
        
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


function [x_hat_hist, tTotal] = getSolutionLiftedHFLSPGOptimized(N,dt,Nt,tol,a,b,mu,dx,x,C,A,F,B,M,phi)

x_hist_sol = zeros(2*N,Nt);
init_q = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
init_w = init_q.^2;
x_hist_sol(1:N,1) = init_q;
x_hist_sol(N+1:2*N,1) = init_w;

n = size(phi,2);
x_hat_hist = zeros(n,Nt);
x_hat_hist(:,1) = phi'*x_hist_sol(:,1);

G = zeros(n,n*n,n);
H = zeros(n,n*n,n);

for i=1:n
    for j=1:n
        H(i,n*(j-1)+i,j) = 1;
    end
    G(i,(i-1)*n+1:i*n,:) = eye(n);
end


F_phi_kron_phi = getF_phi_kron_phi(F,phi);

% LHS
Gamma1 = phi'*A*phi;
Gamma2 = zeros(n,n,n);
Gamma3 = zeros(n,n,n);
Gamma4 = phi'*M*kron(eye(2),phi);
Gamma5 = phi'*A'*phi;
Gamma6 = phi'*A'*A*phi;
Gamma7 = zeros(n,n,n);
Gamma8 = zeros(n,n,n);
Gamma9 = phi'*A'*M*kron(eye(2),phi);
Gamma10 = zeros(n,n,n);
Gamma11 = zeros(n,n,n);
Gamma12 = zeros(n,n,n,n);
Gamma13 = zeros(n,n,n,n);
Gamma14 = zeros(n,n,2*n);
Gamma15 = zeros(n,n,n);
Gamma16 = zeros(n,n,n);
Gamma17 = zeros(n,n,n,n);
Gamma18 = zeros(n,n,n,n);
Gamma19 = zeros(n,n,2*n);
Gamma20 = (kron(eye(2),phi))'*M'*phi;
Gamma21 = (kron(eye(2),phi))'*M'*A*phi;
Gamma22 = zeros(n,2*n,n);
Gamma23 = zeros(n,2*n,n);
Gamma24 = (kron(eye(2),phi))'*M'*M*(kron(eye(2),phi));

% RHS
Gamma25 = phi'*C;
Gamma26 = phi'*A*phi;
Gamma27 = phi'*F_phi_kron_phi;
Gamma28 = phi'*B;
Gamma29 = phi'*M*kron(eye(2),phi);
Gamma30 = phi'*A'*phi;
Gamma31 = phi'*A'*C;
Gamma32 = phi'*A'*A*phi;
Gamma33 = phi'*A'*F_phi_kron_phi;
Gamma34 = phi'*A'*B;
Gamma35 = phi'*A'*M*kron(eye(2),phi);
Gamma36 = zeros(n,n,n);
Gamma37 = zeros(n,n,1);
Gamma38 = zeros(n,n,n);
Gamma39 = zeros(n,n,n*n);
Gamma40 = zeros(n,n,2);
Gamma41 = zeros(n,n,2*n);
Gamma42 = zeros(n,n,n);
Gamma43 = zeros(n,n,1);
Gamma44 = zeros(n,n,n);
Gamma45 = zeros(n,n,n*n);
Gamma46 = zeros(n,n,2);
Gamma47 = zeros(n,n,2*n);
Gamma48 = (kron(eye(2),phi))'*M'*phi;
Gamma49 = (kron(eye(2),phi))'*M'*C;
Gamma50 = (kron(eye(2),phi))'*M'*A*phi;
Gamma51 = (kron(eye(2),phi))'*M'*F_phi_kron_phi;
Gamma52 = (kron(eye(2),phi))'*M'*B;
Gamma53 = (kron(eye(2),phi))'*M'*M*kron(eye(2),phi);



for i=1:n
    % LHS
    Gamma2(i,:,:) = phi'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma3(i,:,:) = phi'*F_phi_kron_phi*squeeze(G(i,:,:));
    Gamma7(i,:,:) = phi'*A'*F_phi_kron_phi*squeeze(H(i,:,:));
    Gamma8(i,:,:) = phi'*A'*F_phi_kron_phi*squeeze(G(i,:,:));
    Gamma10(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*phi;
    Gamma11(i,:,:) = squeeze(H(i,:,:))'*(F_phi_kron_phi)'*A*phi;
    Gamma14(i,:,:) = squeeze(H(i,:,:))'*(F_phi_kron_phi)'*M*(kron(eye(2),phi));
    Gamma15(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*phi;
    Gamma16(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*A*phi;
    Gamma19(i,:,:) = squeeze(G(i,:,:))'*(F_phi_kron_phi)'*M*(kron(eye(2),phi));
    Gamma22(i,:,:) = kron(eye(2),phi)'*M'*(F_phi_kron_phi)*squeeze(H(i,:,:));
    Gamma23(i,:,:) = kron(eye(2),phi)'*M'*(F_phi_kron_phi)*squeeze(G(i,:,:));

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
    Gamma41(i,:,:) = squeeze(H(i,:,:))'*F_phi_kron_phi'*M*kron(eye(2),phi);
    Gamma42(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*phi;
    Gamma43(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*C;
    Gamma44(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*A*phi;
    Gamma45(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*F_phi_kron_phi;
    Gamma46(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*B;
    Gamma47(i,:,:) = squeeze(G(i,:,:))'*F_phi_kron_phi'*M*kron(eye(2),phi);


end

u_kron_x = zeros(2*n,1);

tic;
for t=2:Nt
    x_hat_curr = x_hat_hist(:,t-1);
    x_hat_next = x_hat_hist(:,t-1);

    while true
        u_ = [a*sin(2*pi*(t-1)*dt); b*sin(4*pi*(t-1)*dt)];
        
        x_kron_x = reshape(x_hat_next * x_hat_next', [], 1);
        u_kron_x(1:n) = u_(1)*x_hat_next;
        u_kron_x(n+1:end) = u_(2)*x_hat_next;
        
        u_kron_I = kron(u_,eye(n));

        LHS = eye(n) - dt*Gamma1 - dt*Gamma5 + dt*dt*Gamma6;
        LHS = LHS + (-dt*Gamma4 + dt*dt*Gamma9)*u_kron_I;
        LHS = LHS + u_kron_I'*(-dt*Gamma20 + dt*dt*Gamma21 + dt*dt*Gamma24*u_kron_I);
        
        LHS = LHS + tensorprod(-dt*Gamma2-dt*Gamma3+dt*dt*Gamma7+dt*dt*Gamma8-dt*Gamma10+dt*dt*Gamma11-dt*Gamma15+dt*dt*Gamma16,x_hat_next,1,1);
        LHS = LHS + tensorprod(dt*dt*Gamma14+dt*dt*Gamma19,x_hat_next,1,1)*u_kron_I;
        LHS = LHS + tensorprod(permute(pagemtimes(u_kron_I',dt*dt*permute(Gamma22,[2,3,1])+dt*dt*permute(Gamma23,[2,3,1])),[3,1,2]),x_hat_next,1,1);
        LHS = LHS + tensorprod(tensorprod(dt*dt*(Gamma12+Gamma13+Gamma17+Gamma18),x_hat_next,1,1),x_hat_next,1,1);
        
        
        RHS = x_hat_next - x_hat_curr;
        RHS = RHS + (-dt*Gamma25 + dt*dt*Gamma31);
        RHS = RHS + (-dt*Gamma26 - dt*Gamma30 + dt*dt*Gamma32)*x_hat_next + dt*Gamma30*x_hat_curr;
        RHS = RHS + (-dt*Gamma27 + dt*dt*Gamma33)*x_kron_x;
        RHS = RHS + (-dt*Gamma28 + dt*dt*Gamma34)*u_;
        RHS = RHS + (-dt*Gamma29 + dt*dt*Gamma35)*u_kron_x;

        RHS = RHS + u_kron_I'*((-dt*Gamma48 + dt*dt*Gamma50)*x_hat_next + (dt*Gamma48*x_hat_curr) + (dt*dt*Gamma49));
        RHS = RHS + u_kron_I'*(dt*dt*Gamma51*x_kron_x + dt*dt*Gamma52*u_ + dt*dt*Gamma53*u_kron_x);
        
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
tTotal = toc;

end




function [C, A, F, B, M] = getCoeffMat(N,x,dx,mu)

C = zeros(2*N,1);
A = zeros(2*N,2*N);
F = sparse(2*N,2*N*2*N);
B = zeros(2*N,2);
M = sparse(2*N,2*2*N);

C(N) = mu/dx/dx; %inclusion of the q_{i+1} term on the right boundary for PDE 1
A(2*N,N) = 2*mu/dx/dx; % inclusion of the additional q_i*q_{i+1} term on the right boundary for PDE 2

B(1:N,1) = (1./(1+100*(x-.25).^2))';
B(1:N,2) = (1./(1+100*(x-.75).^2))';


for i=1:N
    A(i,i) = -2/dx/dx*mu;
    if i~=1
        A(i,i-1) = 1/dx/dx*mu;
    end
    if i~=N
        A(i,i+1) = 1/dx/dx*mu;
    end

    F(N+i,2*N*(i-1)+i) = -4*mu/dx/dx;
    if i~=1
        F(N+i,2*N*(i-1)+i-1) = 2*mu/dx/dx;
    end
    if i~=N
        F(N+i,2*N*(i-1)+i+1) = 2*mu/dx/dx;
    end
    
    F(i,2*N*(i-1)+N+i) = -1;
    F(N+i,2*N*N+2*N*(i-1)+N+i) = -2;
    
    M(N+i,i) = 2/(1 + 100*(x(i)-.25).^2);
    M(N+i,2*N+i) = 2/(1 + 100*(x(i)-.75).^2);

    C = sparse(C);
    A = sparse(A);
    F = sparse(F);
    B = sparse(B);
    M = sparse(M);

end


end


function [C, A, F, Fc, B, M] = getCoeffMatNoLift(N,x,dx,mu)

C = zeros(N,1);
A = zeros(N,N);
F = sparse(N,N*N);
B = zeros(N,2);
M = sparse(N,2*N);

C(N) = mu/dx/dx; %inclusion of the q_{i+1} term on the right boundary for PDE 1

B(1:N,1) = (1./(1+100*(x-.25).^2))';
B(1:N,2) = (1./(1+100*(x-.75).^2))';

Fc_list_j = zeros(N,1);

for i=1:N
    A(i,i) = -2/dx/dx*mu;
    if i~=1
        A(i,i-1) = 1/dx/dx*mu;
    end
    if i~=N
        A(i,i+1) = 1/dx/dx*mu;
    end

    Fc_list_j(i) = (N^2+N)*(i-1)+i;

    C = sparse(C);
    A = sparse(A);
    F = sparse(F);
    B = sparse(B);
    M = sparse(M);

end

Fc = sparse(1:N,Fc_list_j,-ones(N,1));

end



function [x_hat_hist, ROMtime] = getECSWLSPGsolutionOptimized(N,dt,Nt,tol,mu,a,b,phi,xi,indices,x)

x = x';
numModes = size(phi,2);
x_hat_hist = zeros(numModes,Nt);

x_init = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hat_hist(:,1) = phi'*x_init;


x_lim = x(indices);
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

        u_sparse = (a*sin(2*pi*(t-1)*dt) ./ (1 + 100*(x_lim-1/4).^2) + b*sin(4*pi*(t-1)*dt) ./ (1 + 100*(x_lim-3/4).^2))';
        
        for i=1:length(indices)
            
            if indices(i)==1
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(0 - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2))) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),2)) = -dt*mu/dx/dx;

            elseif indices(i)==N
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + 1) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),1)) = -dt*mu/dx/dx;
            else
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2))) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),1)) = -dt*mu/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = -dt*mu/dx/dx;
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



function [x_hat_hist, ROMtime] = getECSWGalerkinSolutionOptimized(N,dt,Nt,tol,mu,a,b,phi,xi,indices,x)

x = x';
numModes = size(phi,2);
x_hat_hist = zeros(numModes,Nt);

x_init = x.*(1-x).*(6*(1-x).*(1-x).*exp(-x) - 10*exp(x).*sin(x/6))+x;
x_hat_hist(:,1) = phi'*x_init;

x_lim = x(indices);
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

        u_sparse = (a*sin(2*pi*(t-1)*dt) ./ (1 + 100*(x_lim-1/4).^2) + b*sin(4*pi*(t-1)*dt) ./ (1 + 100*(x_lim-3/4).^2))';


        for i=1:length(indices)
            
            if indices(i)==1
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(0 - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2))) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),2)) = -dt*mu/dx/dx;

            elseif indices(i)==N
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + 1) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),1)) = -dt*mu/dx/dx;

            else
                r_sparse(i) = x_next_sparse(i) - x_curr_sparse(i) - dt*mu/dx/dx*(x_next_sparse_p(neighbor_set(indices(i),1)) - 2*x_next_sparse(i) + x_next_sparse_p(neighbor_set(indices(i),2))) + dt*x_next_sparse(i).^3 - dt*u_sparse(i);
                J_sparse(i,indices_to_indices_p(i)) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next_sparse(i)*x_next_sparse(i);
                J_sparse(i,neighbor_set(indices(i),1)) = -dt*mu/dx/dx;
                J_sparse(i,neighbor_set(indices(i),2)) = -dt*mu/dx/dx;
                
                
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