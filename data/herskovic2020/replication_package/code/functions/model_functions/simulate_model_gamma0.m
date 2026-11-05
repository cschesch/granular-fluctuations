function [table, Sim_frac_hit_constraint, SZ_shr, CorrWS, Model_Output] = simulate_model_gamma0(X,N,Ntop,nloop,loc_exclude,w_tr)

% Paramaters
gamma      = 0; % governs network propagation effect (customer to supplier)
sig0       = X(1); % fundamental shock vol (sigma_epsilon)
sig1       = X(2); % governs internal diversification effect (lambda)
Z          = X(3); % governs prob of new connections
psi        = X(4); % governs sensitivity of weight to customer size
mu_So      = X(5); % mean initial log size distribution
sigma_So   = X(6); % vol initial log size distribution


zetap      = .87;  % governs linkage sparsity effect
nu         = 1;    % governs sensitivity to size of prob of new connection
sig_agg    = 0;    % vol aggregate shocks

if nargin <= 5
    disp('Assuming 10% truncation')
w_tr       = 0.1;  % threshold to calculate w*
end

% Simulation of S
s0 = RandStream('mt19937ar','Seed',0+100); 
RandStream.setGlobalStream(s0);
aux = normrnd(0,1,N,1); 
S   = exp(mu_So + sigma_So * aux); %size distribution S
[~, in_S] = sort(S); % sort vector of firm sizes
TopS = in_S(end-Ntop+1:end); % keep highest-ranked firms


% Allocating Memory
Sim_data_sig       = nan(Ntop,nloop); % v(g) at t+1
Sim_data_Kstar     = nan(Ntop,nloop); % Kstar   at t
Sim_data_Kinstar   = nan(Ntop,nloop); % Kinstar at t
Sim_data_Hstar     = nan(Ntop,nloop); % Histar  at t
Sim_data_Hintildes = nan(Ntop,nloop); % Customer Herfindahl for W tilde star

CorrSV         = nan(1,nloop);
CorrKsS        = nan(1,nloop); % size  and  k-degree*
CorrHsS        = nan(1,nloop); % size  and  Hstar
CorrHsvolHat   = nan(1,nloop); % "Estimated" vol  and  Hstar
CorrKinsS      = nan(1,nloop); % size  and  k-in-degree*
CorrHinsS      = nan(1,nloop); % size  and  Hinstar
CorrHinsvolHat = nan(1,nloop);

CorrWS         = nan(1,nloop);

Sim_data_num_out = nan(1,nloop);
Sim_data_num_in  = nan(1,nloop);
Sim_data_num_net = nan(1,nloop);

Sim_frac_hit_constraint = nan(nloop,3);

parfor loop = 1:nloop
    % Setting the seed
    si = RandStream('mt19937ar','Seed',100+loop);
    RandStream.setGlobalStream(si);
    
    % Draw random connections
    pbase = p_func(S,Z,nu,N,zetap);
    B = binornd(1,pbase);
    B = B - diag(diag(B)); %normalization: a firm has no connection with itself
    K = sum(B,2); % K(i) is the number of costumers of firm i
    
    % Modify B to have firms with at least one link:
    in_K0 = find(K==0); % indices of firms with 0 customers
    rp = randsample(N-1,length(in_K0),'true');
    for ff=1:length(in_K0)
        if rp(ff)>=in_K0(ff)
            rp(ff) = rp(ff) + 1; % don't +1 in your own column
        end
        B(in_K0(ff),rp(ff)) = 1; % create one connection for those suppliers (rows) that didnt have any customers (columns)
    end
    K = sum(B,2); % K(i) is the number of costumers of firm i
    
    %  ------------------------ DEFINING MATRIX W -----------------------------
    W = B.*repmat(S'.^psi, N, 1);
    div = sum(W,2);
    div(div==0) = 1;
    W = bsxfun(@rdivide,W,div); % normalize weight rows to sum to 1
    
    %  ---------------------- MATRIX INVERSION STEP --------------------------------
    %Warning
    if sum(sum(W*0==0))<N^2
        disp('W Problem');
        W(1:10,1:10)
        B(1:10,1:10)
        S(1:10)
        K(1:10)
        div(1:10)
        pbase(1:10,1:10)
        kjbcdisbcn
        
    else
        
    %%% Matrix inversion [code spends most time here]
    invmat = (eye(N) - gamma*W)\eye(N);
        
    % New sigma_i function
    sig = sig0*ones(N,1) + sig1*log(1 + exp(mu_So)./S);

    Sim_frac_hit_constraint(loop,:) = [mean(reshape(pbase,[],1) == 0), ...
                                      mean(reshape(pbase,[],1) == 1), ...
                                      mean(sig == .01)];

    % Firms' Volatility
    Sigma_E = sig_agg^2*ones(N) + diag(sig.^2); %adding aggregate volatility
    Vg    = invmat * Sigma_E * invmat'; %varcov matrix of growth rates (eqn 6)
    sigi2 = diag(Vg); %individual variances of growth rates
        
    %  ---------------------- CALCULATING NETWORK MOMENTS -----------------
    % Calculating the Truncated W and B matrices
    in_star = (W>=w_tr); % indices for w threshold truncation (if model not 4)
    W = W.*in_star;
    W_dollar = bsxfun(@times,W,S); % Dollar weighted

    % Truncation (CRSP-Compustat selection)
    W = W(TopS,TopS); 
    W_dollar = W_dollar(TopS,TopS); 

    Kstar   = sum(W>0,2);   % truncated out-degree - supplier degree
    Kinstar = sum(W>0,1)';  % truncated in-degree  - customer degree

    % Herfindahl (Hout)
    Histar  = sum(W.^2,2);  % truncated Herfindahl (supplier)
    Histar(Histar == 0) = nan;

    % Herfindahl (Hin)
    Sum_Waux = sum(W_dollar,1); Sum_Waux(Sum_Waux==0) = 1;
    W_in = bsxfun(@times, W_dollar, 1 ./ Sum_Waux);

    Hintildes = sum(W_in.^2,1)';
    Hintildes(Hintildes == 0) = nan;

    % Locating firms
    loc_out = Kstar   >= 1;% set of compustat segment firms with network data
    loc_in  = Kinstar >= 1; % set of compustat segment firms with network data
    loc_net = loc_in | loc_out;

    % Counting
    num_out = sum(loc_out);
    num_in  = sum(loc_in);
    num_net = sum(loc_net);

    %  ---------------------- SAVING VARIABLES --------------------------------
    Sim_data_num_out(loop) = num_out;
    Sim_data_num_in(loop)  = num_in;
    Sim_data_num_net(loop) = num_net;

    Sim_data_sig(:,loop)       = log(sigi2(TopS)); % v(g) at t+1
    Sim_data_Kstar(:,loop)     = Kstar;   % Kstar   at t
    Sim_data_Kinstar(:,loop)   = Kinstar; % Kinstar at t
    Sim_data_Hstar(:,loop)     = Histar;  % Histar  at t
    Sim_data_Hintildes(:,loop) = Hintildes; % Customer Herfindahl for W tilde star

    if num_net > 1
        CorrSV(loop)           = corr(log(sigi2(TopS(loc_net))),log(S(TopS(loc_net))));
    end

    if num_out > 1
        CorrKsS(loop)          = corr(log(Kstar(loc_out)),log(S(TopS(loc_out))));      % size  and  k-degree*
        CorrHsS(loop)          = corr(log(Histar(loc_out)),log(S(TopS(loc_out))));      % size  and  Hstar
        CorrHsvolHat(loop)     = corr(log(Histar(loc_out)),log(sigi2(TopS(loc_out))));        % "Estimated" vol  and  Hstar
    end

    if num_in > 1
        CorrKinsS(loop)        = corr(log(Kinstar(loc_in)),log(S(TopS(loc_in))));    % size  and  k-in-degree*
        CorrHinsS(loop)        = corr(log(Hintildes(loc_in)),log(S(TopS(loc_in))));      % size  and  Hinstar
        CorrHinsvolHat(loop)   = corr(log(Hintildes(loc_in)),log(sigi2(TopS(loc_in))));        % "Estimated" vol  and  Hinstar
    end
        
    nanW = W; 
    nanW(W == 0) = nan; 
    nanW(~loc_out,:) = nan;
   % CorrWS(loop) = nanmean(corr(nanW', log(S(TopS)),'rows','pairwise'))
    CorrWS(loop) = nanmean(corr(log(nanW'), log(S(TopS)),'rows','pairwise'))
    
    end
end
Sim_data_S   = repmat(S(TopS),1,nloop);    % size at t

Model_Output.Sim_data_S       = Sim_data_S;
Model_Output.Sim_data_sig     = Sim_data_sig;
Model_Output.Sim_data_Kstar   = Sim_data_Kstar;
Model_Output.Sim_data_Kinstar = Sim_data_Kinstar;

%% Saving Results
pct     = [5 10 25 50 75 90 95];

% identifying firms without network data
Sim_data_Kstar(Sim_data_Kstar == 0)         = nan;
Sim_data_Kinstar(Sim_data_Kinstar == 0)     = nan;

% Size and vol distribution: all firms
SZ = Sim_data_S;    SZ_agg = nansum(SZ,1);
SG = Sim_data_sig;
tables.Table_Size = [(nanmean(log(SZ))); (nanstd(log(SZ))); (prctile(log(SZ),pct))];
tables.Table_Vol  = [(nanmean(SG)); (nanstd(SG)); (prctile(SG,pct)); (CorrSV)];

% Size and vol distribution: supplier with network data
SZ = Sim_data_S;   SZ(isnan(Sim_data_Kstar))       = nan;
SG = Sim_data_sig; SG(isnan(Sim_data_Kstar))       = nan;
tables.Table_Size_SUPP = [(nanmean(log(SZ))); (nanstd(log(SZ))); (prctile(log(SZ),pct))];
tables.Table_Vol_SUPP  = [(nanmean(SG)); (nanstd(SG)); (prctile(SG,pct)); (CorrSV)];

% Size and vol distribution: customer with network data
SZ = Sim_data_S;   SZ(isnan(Sim_data_Kinstar))       = nan;
SG = Sim_data_sig; SG(isnan(Sim_data_Kinstar))       = nan;
tables.Table_Size_CUST = [(nanmean(log(SZ))); (nanstd(log(SZ))); (prctile(log(SZ),pct))];
tables.Table_Vol_CUST  = [(nanmean(SG)); (nanstd(SG)); (prctile(SG,pct)); (CorrSV)];

% Size and vol distribution: firms with some network data
SZ = Sim_data_S;   SZ(isnan(Sim_data_Kstar) & isnan(Sim_data_Kinstar))         = nan;
SG = Sim_data_sig; SG(isnan(Sim_data_Kstar) & isnan(Sim_data_Kinstar))       = nan;
tables.Table_Size_NETW = [(nanmean(log(SZ))); (nanstd(log(SZ))); (prctile(log(SZ),pct))];
tables.Table_Vol_NETW  = [(nanmean(SG)); (nanstd(SG)); (prctile(SG,pct)); (CorrSV)];
SZ_shr = nansum(SZ,1) ./ SZ_agg;
SZ_shr = nanmean(SZ_shr);

% Network moments 
Sim_data_Hstar(isnan(Sim_data_Kstar))         = nan;
Sim_data_Hintildes(isnan(Sim_data_Kinstar))   = nan;
        
tables.Table_Net_Mom_A = [(prctile(Sim_data_Kstar,pct)); ...
    (prctile(Sim_data_Hstar,pct)); ...
    (CorrKsS); (CorrHsS); (CorrHsvolHat)];
tables.Table_Net_Mom_B = [(prctile(Sim_data_Kinstar,pct)); ...
    (prctile(Sim_data_Hintildes,pct)); ...
    (CorrKinsS); (CorrHinsS); (CorrHinsvolHat)];

table = [tables.Table_Size_NETW;...
         tables.Table_Vol_NETW;...
         tables.Table_Net_Mom_A;...
         tables.Table_Net_Mom_B;...
         (Sim_data_num_net)];
     
table(loc_exclude,:) = [];
