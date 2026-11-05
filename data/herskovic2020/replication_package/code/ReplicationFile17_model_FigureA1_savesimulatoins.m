clear, addpath(genpath('functions\'))

disp(' ')
disp(' ')
disp('---------------------------------------------------------------------')
disp('        PLOTTING FIGURES')
disp('---------------------------------------------------------------------')
disp(' ')
disp(' ') 
rng(0)
SIM = 50000;
k   =  1000;

TABall_hfd  = nan(k,SIM);
TABall_var  = nan(k,SIM);
TABall_var1 = nan(k,SIM);
TABall_var2 = nan(k,SIM);
TABall_ConsShare = nan(SIM,1);

L=(exp(linspace(0,1,k))-1)/(exp(1)-1);
L=1+L*2;

parfor sim = 1:SIM
    if floor(sim/100)*100==sim, disp(['simulation: ' num2str(sim)]), end
    n = 3;
    ok=true;
    while ok
        A=rand(n);
        if max(sum(A,2)) > 1, continue; end                          % condition i
        [n,~]=size(A);
        theta = (eye(n) - A') * ones(n,1); theta = theta/sum(theta); % theta such that firms have the same size
        if min(theta) >= 0 && max(sum(A,2)) <= 1, ok = false; end    % condition ii
    end

    psi = inv(eye(n) - A') * theta;
    delta = psi/sum(psi);
    theta0=theta;

    % Computing Consumption Share
    delta = delta/sum(delta);
    [n,~]=size(A);
    B = ones(n,1) - sum(A,2);

    psi = delta / sum((eye(n) - A') * delta);
    theta = (eye(n) - A') * psi;

    Wt = diag(psi) \ A' * diag(psi);
    ConShr = 1/sum(psi);
    TABall_ConsShare(sim) = ConShr;

    % Normalizing aggregate variance
    rho = -1 / (n-1);
    S1 = rho*ones(n) + (1-rho) * eye(n);
    Mt = inv((eye(n) - A')) * S1 * inv((eye(n) - A));
    sig2 = 1;
    S0 = sig2 * S1;
    M = sig2*Mt;

    psi = delta / sum((eye(n) - A') * delta);
    theta = (eye(n) - A') * psi;

    Wt = diag(psi) \ A' * diag(psi);
    S = inv(diag(psi)) * S0 * inv(diag(psi));
    V = inv(eye(n)-Wt) * S * inv(eye(n)-Wt)';
    psi_bar = mean(1./psi);
    D = 1./psi - psi_bar;

    psi1 = psi(1);
    psi2 = psi(2);
    psi3 = psi(3);


    avg_var0 = mean(diag(V));

    aTABall_hfd  = nan(k,1);
    aTABall_var  = nan(k,1);
    aTABall_var1 = nan(k,1);
    aTABall_var2 = nan(k,1);


    % Loop over different levels of size Herfindahl
    for iL=1:k
        scale=L(iL);        
        rho = -1 / (n-1);
        S1 = rho*ones(n) + (1-rho) * eye(n);
        Mt = inv((eye(n) - A')) * S1 * inv((eye(n) - A));
        sig2 = 1;
        S0 = sig2 * S1;
        M  = sig2 * Mt;
        H = sum((ones(n,1)/n).^2) * scale;

        %%% Using Closed form solution for theta to match size Herfindahl
% For a given level of Herfindahl, we find a vector of preference parameters theta that 
%   generates that level of Herfindahl in equilibrium (the equilibrium is unique). In 
%   equilibrium, Herfindahl is a quadratic function of thetas (it's a sum of squared shares, 
%   which in turn depend on theta). So there might be two solutions --- that is, two distinct 
%   configurations of preference parameters that deliver the same size Herfindahl. 
%   We solve explicitly for the two possible solutions. To be conservative, we keep 
%   the solution (vector theta) that delivers the lowest average variance in equilibrium.
        At=inv(eye(n)-A');
        bt=sum(At,1)';
        Ct=nan(size(At)); for j=1:n; for s=1:n; Ct(j,s)=sum(At(:,j).*At(:,s)); end; end
        tss = theta(1:end-3);
        aA = (1/ConShr - bt(1:end-3)'* tss - bt(end) * (1 - sum(tss))) / (bt(end-1)-bt(end));
        aB = ( - bt(end-2) +  bt(end)) / (bt(end-1)-bt(end));
        aC = 1 - sum(tss) - aA;
        aD = - (1 + aB);
        tn1 = @(tn2) aA + aB * tn2;
        tn  = @(tn2) aC + aD * tn2;

        Ct11 = Ct(1:end-3,1:end-3);
        Ct12 = Ct(1:end-3,end-2:end);
        Ct21 = Ct(end-2:end,1:end-3);
        Ct22 = Ct(end-2:end,end-2:end);
        alph1 = [0; aA; aC];
        alph2 = [1; aB; aD];
        chat = -H * (ConShr^(-2))+ tss'*Ct(1:end-3,1:end-3)*tss + [0; aA; aC]' * Ct22 * [0; aA; aC];
        bhat = alph1'* Ct22 * alph2 + alph2'* Ct22 * alph1;
        ahat = alph2'*Ct22*alph2;

        dhat = bhat^2 - 4 * ahat * chat;
        if dhat < 0
            continue
        end

        tn2_sol1 = (-bhat + sqrt(max(dhat,0))) / (2 * ahat);
        tn2_sol2 = (-bhat - sqrt(max(dhat,0))) / (2 * ahat);

        TAB_var = [];
        for solloop = [1 2]
            switch solloop
                case 1, tn2 = tn2_sol1;
                case 2, tn2 = tn2_sol2;
            end
            theta = [tss, tn2; tn1(tn2); tn(tn2)]; % vector of preference parameters
            psi = inv(eye(n) - A') * theta;
            delta = psi ./ sum(psi);
            Wt = diag(psi) \ A' * diag(psi);
            rho = -1 / (n-1);
            S1 = rho*ones(n) + (1-rho) * eye(n);
            Mt = inv((eye(n) - A')) * S1 * inv((eye(n) - A));
            sig2 = 1;
            S0 = sig2 * S1;
            M = sig2*Mt;
            Wt = diag(psi) \ A' * diag(psi);
            S = inv(diag(psi)) * S0 * inv(diag(psi));
            V = inv(eye(n)-Wt) * S * inv(eye(n)-Wt)';
            avg_var = mean(diag(V));
            if min(theta)<0, avg_var = nan; end
            TAB_var = [TAB_var; avg_var];
        end
        aTABall_var1(iL) =  log(TAB_var(1)/avg_var0);
        aTABall_var2(iL) =  log(TAB_var(2)/avg_var0);

        if numel(TAB_var) == sum(isnan(TAB_var)), continue, end
        % Keeping the solution with minimum variance
        sol = find(TAB_var == min(TAB_var),1);
        switch sol
            case 1, tn2 = tn2_sol1;
            case 2, tn2 = tn2_sol2;
        end
        theta = [tss, tn2; tn1(tn2); tn(tn2)];
        psi = inv(eye(n) - A') * theta;

        delta = psi ./ sum(psi);

        Wt = diag(psi) \ A' * diag(psi);

        % Normalizing aggregate variance
        rho = -1 / (n-1);
        S1 = rho*ones(n) + (1-rho) * eye(n);
        Mt = inv((eye(n) - A')) * S1 * inv((eye(n) - A));
        sig2 = 1;
        S0 = sig2 * S1;
        M = sig2*Mt;

        Wt = diag(psi) \ A' * diag(psi);
        S = inv(diag(psi)) * S0 * inv(diag(psi));
        V = inv(eye(n)-Wt) * S * inv(eye(n)-Wt)';

        avg_var = mean(diag(V));
        theta = (eye(n) - A') * psi;
        aTABall_hfd(iL) =  sum(delta.^2);
        aTABall_var(iL) =  log(avg_var/avg_var0);
    end

    TABall_hfd(:,sim)  = aTABall_hfd;
    TABall_var(:,sim)  = aTABall_var;
    TABall_var1(:,sim) = aTABall_var1;
    TABall_var2(:,sim) = aTABall_var2;

end

save('../output/FigureA1simulations/FigureA1_simulation.mat','TABall*','L')
