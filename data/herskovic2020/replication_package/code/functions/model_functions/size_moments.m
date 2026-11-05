function [out] = size_moments(XSz,X,N,Ntop,nloop)

X(6) = XSz(1);
X(7) = exp(XSz(2));

Z          = X(4);%governs prob of new connections
psi        = X(5); %governs sensitivity of weight to customer size
mu_So      = X(6);%15.40; %15.5mean initial log size distribution
sigma_So   = X(7);%1.32;%1.33vol initial log size distribution


zetap      = .87; %.87;%governs linkage sparsity effect
nu         = 1;   %governs sensitivity to size of prob of new connection
w_tr        = 0.1; % threshold to calculate w*

s0 = RandStream('mt19937ar','Seed',0+100);
RandStream.setGlobalStream(s0);
aux = normrnd(0,1,N,1); 

S = exp(mu_So + sigma_So * aux); %size distribution S
[~, in_S] = sort(S); % sort vector of firm sizes
TopS = in_S(end-Ntop+1:end); % keep highest-ranked firms

% Simulate
Sim_data_Kstar     = nan(Ntop,nloop);% Kstar   at t
Sim_data_Kinstar   = nan(Ntop,nloop);% Kinstar at t


parfor loop = 1:nloop
    loop;
    si = RandStream('mt19937ar','Seed',100+loop);
    RandStream.setGlobalStream(si);
    
    pbase = p_func(S,Z,nu,N,zetap);
    B = binornd(1,pbase);
    B = B - diag(diag(B)); %normalization: a firm has no connection with itself
    K = sum(B,2); % K(i) is the number of costumers of firm i
    
    in_K0 = find(K==0); % indices of firms with 0 customers
    rp = randsample(N-1,length(in_K0),'true');
    for ff=1:length(in_K0)
        if rp(ff)>=in_K0(ff)
            rp(ff) = rp(ff) + 1; % don't +1 in your own column
        end
        B(in_K0(ff),rp(ff)) = 1; % create one connection for those suppliers (rows) that didnt have any customers (columns)
    end
    
    W = B.*repmat(S'.^psi, N, 1);
    div = sum(W,2); % sum weights for normalization below
    div(div==0) = 1;
    W = bsxfun(@rdivide,W,div); % normalize weight columns to sum to 1
    
    if sum(sum(W*0==0))<N^2
        disp('W Problem');
    else
        
        in_star = (W>=w_tr); % indices for w threshold truncation (if model not 4)
        W = W.*in_star;
        W = W(TopS,TopS); % Truncation (CRSP-Compustat selection)
        
        Kstar   = sum(W>0,2);   % truncated out-degree - supplier degree
        Kinstar = sum(W>0,1)';  % truncated in-degree  - customer degree
        
        Sim_data_Kstar(:,loop)     = Kstar;   % Kstar   at t
        Sim_data_Kinstar(:,loop)   = Kinstar; % Kinstar at t

    end
end
SZ   = repmat(S(TopS),1,nloop);    % size at t

% identifying firms without network data
Sim_data_Kstar(Sim_data_Kstar == 0)         = nan;
Sim_data_Kinstar(Sim_data_Kinstar == 0)     = nan;

% Size and vol distribution: firms with some network data
SZ(isnan(Sim_data_Kstar) & isnan(Sim_data_Kinstar))         = nan;

XS_AvgSz = nanmean(nanmean(log(SZ)));
XS_StdSz = nanmean(nanstd(log(SZ)));

out = [XS_AvgSz, XS_StdSz]';
disp([XS_AvgSz, XS_StdSz, mu_So, sigma_So])

