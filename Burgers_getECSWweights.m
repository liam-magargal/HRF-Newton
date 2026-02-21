clc;
clear all;
close all;

% user inputs
N = 1024;
dt = .001;
Nt = 500;

Nh = 5000; % number of sample solutions used to get the ECSW weights (following the procedure of Grimberg, 2020)

dx = 1/(N+1);
x = linspace(dx,1-dx,N);
tol = 1e-6;

t_domain = linspace(0,dt*Nt,Nt);
[X, T] = meshgrid(x,t_domain);

x_hist = getAllTrainingSolutionsOptimized(N,dt,Nt,tol,x,dx);


%% ROMs
[U,Sigma,V] = svd(x_hist,'econ');

latTol = [10^(-2/2) 10^(-3/2) 10^(-4/2) 10^(-5/2) 10^(-6/2) 10^(-7/2) 10^(-8/2)]';

uStan_error = zeros(N,1);

for i=1:N
    uStan_error(i) = 1-sum(diag(Sigma(1:i,1:i)).^2) / sum(diag(Sigma).^2);
end


for i=1:size(latTol,1)
    i
    for j=1:size(uStan_error,1)
        if uStan_error(j)<latTol(i)
            numModes = j;
            break
        end
    end


    phi = U(:,1:numModes);

    
    % for ECSW solutions
    numSol = 10;

    e_ecsw = 1e-5;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/LSPG_1e5_', int2str(i), '.mat'),'xi','indices')

    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/Galerkin_1e5_', int2str(i), '.mat'),'xi','indices')

    e_ecsw = 1e-7;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/LSPG_1e7_', int2str(i), '.mat'),'xi','indices')

    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/Galerkin_1e7_', int2str(i), '.mat'),'xi','indices')

    e_ecsw = 1e-9;
    [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/LSPG_1e9_', int2str(i), '.mat'),'xi','indices')
    
    [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x);
    save(strcat('BurgersECSWweights/Galerkin_1e9_', int2str(i), '.mat'),'xi','indices')

  
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




function [xi, indices] = getECSWLSPGweightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x)

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
        mu_left = 1+ .25*(l-1);
        for m=1:numSol
            mu2 = .01 + .01*(m-1);
            for i=1:Nt-1
                for k=0:1
                    if ismember((l-1)*Nt*numSol*numSol+(m-1)*Nt*numSol+(2*i+k-1),sampleStates)
                        x_curr = phi*x_hat_hist(:,i);
                        x_next = phi*x_hat_hist(:,i+k);
       

                        r = zeros(Ne,1);
                        J = zeros(Ne,Ne);
                
                        r(1) = x_next(1) - x_curr(1) + dt/(2*dx)*x_next(1)*(x_next(2) - mu_left) - dt*mu2/dx/dx*(x_next(2) - 2*x_next(1) + mu_left);
                        r(Ne) = x_next(Ne) - x_curr(Ne) + dt/(2*dx)*x_next(Ne)*(0 - x_next(Ne-1)) - dt*mu2/dx/dx*(0 - 2*x_next(Ne) + x_next(Ne-1));
                        r(2:Ne-1) = x_next(2:Ne-1) - x_curr(2:Ne-1) + dt/(2*dx)*x_next(2:Ne-1).*(x_next(3:Ne) - x_next(1:Ne-2)) - dt*mu2/dx/dx*(x_next(1:Ne-2) - 2*x_next(2:Ne-1) + x_next(3:Ne));
                        
                        J(1,1) = 1 + dt/2/dx*(x_next(2)-mu_left) + 2*dt*mu2/dx/dx;
                        J(1,2) = dt/2/dx*x_next(1) - mu2*dt/dx/dx;
                
                        J(Ne,Ne) = 1 + dt/2/dx*(0-x_next(Ne-1)) + 2*dt*mu2/dx/dx;
                        J(Ne,Ne-1) = -dt/2/dx*x_next(Ne) - mu2*dt/dx/dx;
                
                        for j=2:Ne-1
                            J(j,j) = 1 + dt/2/dx*(x_next(j+1)-x_next(j-1)) + 2*dt*mu2/dx/dx;
                            J(j,j-1) = -dt/2/dx*x_next(j) - mu2*dt/dx/dx;
                            J(j,j+1) = dt/2/dx*x_next(j) - mu2*dt/dx/dx;
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

function [xi, indices] = getECSWGalerkinWeightsTwoStepSampledSol(phi,x_hist,dt,Nt,numSol,e_ecsw,Nh,x)

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
        mu_left = 1+ .25*(l-1);
        for m=1:numSol
            mu2 = .01 + .01*(m-1);
            for i=1:Nt-1
                for k=0:1
                    if ismember((l-1)*Nt*numSol*numSol+(m-1)*Nt*numSol+(2*i+k-1),sampleStates)
                        x_curr = phi*x_hat_hist(:,i);
                        x_next = phi*x_hat_hist(:,i+k);
       

                        r = zeros(Ne,1);
                        J = zeros(Ne,Ne);
                
                        r(1) = x_next(1) - x_curr(1) + dt/(2*dx)*x_next(1)*(x_next(2) - mu_left) - dt*mu2/dx/dx*(x_next(2) - 2*x_next(1) + mu_left);
                        r(Ne) = x_next(Ne) - x_curr(Ne) + dt/(2*dx)*x_next(Ne)*(0 - x_next(Ne-1)) - dt*mu2/dx/dx*(0 - 2*x_next(Ne) + x_next(Ne-1));
                        r(2:Ne-1) = x_next(2:Ne-1) - x_curr(2:Ne-1) + dt/(2*dx)*x_next(2:Ne-1).*(x_next(3:Ne) - x_next(1:Ne-2)) - dt*mu2/dx/dx*(x_next(1:Ne-2) - 2*x_next(2:Ne-1) + x_next(3:Ne));
                        
                        J(1,1) = 1 + dt/2/dx*(x_next(2)-mu_left) + 2*dt*mu2/dx/dx;
                        J(1,2) = dt/2/dx*x_next(1) - mu2*dt/dx/dx;
                
                        J(Ne,Ne) = 1 + dt/2/dx*(0-x_next(Ne-1)) + 2*dt*mu2/dx/dx;
                        J(Ne,Ne-1) = -dt/2/dx*x_next(Ne) - mu2*dt/dx/dx;
                
                        for j=2:Ne-1
                            J(j,j) = 1 + dt/2/dx*(x_next(j+1)-x_next(j-1)) + 2*dt*mu2/dx/dx;
                            J(j,j-1) = -dt/2/dx*x_next(j) - mu2*dt/dx/dx;
                            J(j,j+1) = dt/2/dx*x_next(j) - mu2*dt/dx/dx;
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
