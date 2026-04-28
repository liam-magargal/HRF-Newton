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

dx = 1/(N+1);
x = linspace(dx,1-dx,N);
tol = 1e-6;
a_all = [-2 -1 0 1 2];
b_all = [0 -2 1 -1 2];
t_domain = linspace(0,dt*Nt,Nt+1);
[X, T] = meshgrid(x,t_domain);


x_hist_sol1 = getSolutionOptimizedNoLift(N,dt,Nt+1,tol,-2,0,mu,dx,x);
x_hist_sol2 = getSolutionOptimizedNoLift(N,dt,Nt+1,tol,-1,-2,mu,dx,x);
x_hist_sol3 = getSolutionOptimizedNoLift(N,dt,Nt+1,tol,0,1,mu,dx,x);
x_hist_sol4 = getSolutionOptimizedNoLift(N,dt,Nt+1,tol,1,-1,mu,dx,x);
x_hist_sol5 = getSolutionOptimizedNoLift(N,dt,Nt+1,tol,2,2,mu,dx,x);
x_hist = [x_hist_sol1 x_hist_sol2 x_hist_sol3 x_hist_sol4 x_hist_sol5];


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

    e_ecsw = 1e-5;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/LSPG_1e5_', int2str(i), '.mat'),'xi','indices')

    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/Galerkin_1e5_', int2str(i), '.mat'),'xi','indices')

    e_ecsw = 1e-7;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/LSPG_1e7_', int2str(i), '.mat'),'xi','indices')

    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/Galerkin_1e7_', int2str(i), '.mat'),'xi','indices')


    e_ecsw = 1e-9;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/LSPG_1e9_', int2str(i), '.mat'),'xi','indices')

    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phiStan,x_hist,dt,Nt+1,numSol,e_ecsw,a_all,b_all,mu,Nh,x);
    save(strcat('HeatDiffECSWweights/Galerkin_1e9_', int2str(i), '.mat'),'xi','indices')
    
  
end



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



function [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,a_all,b_all,mu,Nh,x)

    x_hat_hist = phi'*x_hist;
    Ne = size(phi,1);
    n = size(phi,2);

    dx = x(2)-x(1);

    allStates = randperm(2*(Nt-1)*numSol*numSol);
    sampleStates = allStates(1:Nh);

    C = zeros(Nh*n,Ne);
    d = zeros(Nh*n,1);

    count = 0;
    for l=1:numSol
        for m=1:numSol
            a = a_all(m);
            b = b_all(m);
            for i=1:Nt-1
                for k=0:1
                    if ismember((l-1)*Nt*numSol*numSol+(m-1)*Nt*numSol+(2*i+k-1),sampleStates)
                        x_curr = phi*x_hat_hist(:,i);
                        x_next = phi*x_hat_hist(:,i+k);
       
                        r = zeros(Ne,1);
                        J = zeros(Ne,Ne);

                        u = (a*sin(2*pi*(i-1)*dt) ./ (1 + 100*(x-1/4).^2) + b*sin(4*pi*(i-1)*dt) ./ (1 + 100*(x-3/4).^2))';
                        r(1) = x_next(1) - x_curr(1) - dt*mu/dx/dx*(0 - 2*x_next(1) + x_next(2)) + dt*x_next(1).^3 - dt*u(1);
                        r(end) = x_next(end) - x_curr(end) - dt*mu/dx/dx*(x_next(end-1) - 2*x_next(end) + 1) + dt*x_next(end).^3 - dt*u(end);
                        r(2:end-1) = x_next(2:end-1) - x_curr(2:end-1) - dt*mu/dx/dx*(x_next(1:end-2) - 2*x_next(2:end-1) + x_next(3:end)) + dt*x_next(2:end-1).^3 - dt*u(2:end-1);
                        
                        J(1,1) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(1)^2;
                        J(1,2) = - dt*mu/dx/dx;
                        
                        J(end,end) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(end)^2;
                        J(end,end-1) = - dt*mu/dx/dx;
                
                
                        for j=2:Ne-1
                            J(j,j) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next(j)*x_next(j);
                            J(j,j-1) = -dt*mu/dx/dx;
                            J(j,j+1) = -dt*mu/dx/dx;
                        end
            
                        W = J*phi;
                        for j=1:Ne
                            C(count*n+1:(count+1)*n,j) = W(j,:)'*r(j);
                            d(count*n+1:(count+1)*n) = d(count*n+1:(count+1)*n) + C(count*n+1:(count+1)*n,j);
                        end
                        count = count + 1;
                    end
                end
            end
        end
    end
   
    tol = e_ecsw;
    options = optimset('TolX',tol);
    xi_full = lsqnonneg(C,d,options);
    indices = (xi_full~=0);
    indices = find(indices);
    xi = xi_full(indices);
    indicies = indices';

    if ~ismember(1,indices)
        indices = [indices; 1];
    end
    CtC = C(:,indices)'*C(:,indices);
    xi = CtC\C(:,indices)'*d;

    indices_p = indices;

end




function [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,a_all,b_all,mu,Nh,x)

    x_hat_hist = phi'*x_hist;
    Ne = size(phi,1);
    n = size(phi,2);

    dx = x(2)-x(1);
    
    allStates = randperm(2*(Nt-1)*numSol*numSol);
    sampleStates = allStates(1:Nh);

    C = zeros(Nh*n,Ne);
    d = zeros(Nh*n,1);

    count = 0;
    for l=1:numSol
        for m=1:numSol
            a = a_all(m);
            b = b_all(m);
            for i=1:Nt-1
                for k=0:1
                    if ismember((l-1)*Nt*numSol*numSol+(m-1)*Nt*numSol+(2*i+k-1),sampleStates)
                        x_curr = phi*x_hat_hist(:,i);
                        x_next = phi*x_hat_hist(:,i+k);
       
                        r = zeros(Ne,1);
                        J = zeros(Ne,Ne);

                        u = (a*sin(2*pi*(i-1)*dt) ./ (1 + 100*(x-1/4).^2) + b*sin(4*pi*(i-1)*dt) ./ (1 + 100*(x-3/4).^2))';
                        r(1) = x_next(1) - x_curr(1) - dt*mu/dx/dx*(0 - 2*x_next(1) + x_next(2)) + dt*x_next(1).^3 - dt*u(1);
                        r(end) = x_next(end) - x_curr(end) - dt*mu/dx/dx*(x_next(end-1) - 2*x_next(end) + 1) + dt*x_next(end).^3 - dt*u(end);
                        r(2:end-1) = x_next(2:end-1) - x_curr(2:end-1) - dt*mu/dx/dx*(x_next(1:end-2) - 2*x_next(2:end-1) + x_next(3:end)) + dt*x_next(2:end-1).^3 - dt*u(2:end-1);
                        
                        J(1,1) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(1)^2;
                        J(1,2) = - dt*mu/dx/dx;
                        
                        J(end,end) = 1 + 2*dt*mu/dx/dx + dt*3*x_next(end)^2;
                        J(end,end-1) = - dt*mu/dx/dx;
                
                
                        for j=2:Ne-1
                            J(j,j) = 1 + 2*dt*mu/(dx*dx) + 3*dt*x_next(j)*x_next(j);
                            J(j,j-1) = -dt*mu/dx/dx;
                            J(j,j+1) = -dt*mu/dx/dx;
                        end
            
                        W = phi;
                        for j=1:Ne
                            C(count*n+1:(count+1)*n,j) = W(j,:)'*r(j);
                            d(count*n+1:(count+1)*n) = d(count*n+1:(count+1)*n) + C(count*n+1:(count+1)*n,j);
                        end
                        count = count + 1;
                    end
                end
            end
        end
    end
   
    tol = e_ecsw;
    options = optimset('TolX',tol);
    xi_full = lsqnonneg(C,d,options);
    indices = (xi_full~=0);
    indices = find(indices);
    xi = xi_full(indices);
    indicies = indices';

    if ~ismember(1,indices)
        indices = [indices; 1];
    end
    CtC = C(:,indices)'*C(:,indices);
    xi = CtC\C(:,indices)'*d;

    indices_p = indices;

end
