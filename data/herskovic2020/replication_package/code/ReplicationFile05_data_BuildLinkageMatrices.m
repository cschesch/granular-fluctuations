clear, addpath(genpath('functions\'))

%**************************************************************************
% Load linkages2013.mat.  This is our extension of the Cohen-Frazzini
% data through 2013.  This only includes cust/supp pairs for which both
% gvkeys are available, so is inferior to the raw segment file in its
% information regarding customer counts and customer weights.  However, it
% is the best we can do for supplier counts and weights.
%**************************************************************************

linkages    = load('../data/linkages2013');
linkages.ctotsale(linkages.ctotsale<=0) = nan;
linkages.percsale(linkages.percsale<=0) = nan;
linkages.sale(linkages.sale<=0)         = nan;

% Impose 10% truncation
linkages.percsale(linkages.percsale<.1) = nan;

%**************************************************************************
% Build matrices
%**************************************************************************

%permlist    = unique(linkages.permno); 
%%%%%%% comment %%%%%%%
%       PERMLIST SHOULD INCLUDE PERMNO OF CUSTOMERS: -- this adds about 2k firms
permlist    = unique([linkages.permno; linkages.cpermno]);
%fileID      = fopen('../data/permlist.txt','w');
%fprintf(fileID,'%6.0f\n',permlist);
yrlist      = unique(linkages.dates);
N           = length(permlist);
T           = length(yrlist);

% Build matrices of customer information (for supplier i)
% Customer-supplier info is fiscal year
CUSTNUM     = nan(T,N);
CUSTOUT     = nan(T,N);
CUSTHERF    = nan(T,N);
CUSTSZWT    = nan(T,N);
for i=1:N
    loci            = find(linkages.permno==permlist(i));
    for t=1:T
        loct        = find(linkages.dates(loci)==yrlist(t));
        if isempty(loct), continue, end
        loc             = loci(loct);
        wt              = linkages.percsale(loc);
        CUSTNUM(t,i)    = sum(~isnan(wt));
        CUSTOUT(t,i)    = nansum( wt );
        CUSTHERF(t,i)   = nansum( wt.^2 ); clear wt
        
        tmp             = [linkages.percsale(loc),linkages.ctotsale(loc)];
        if sum(~isnan(sum(tmp,2)))>2
            %tmp1            = corr( tmp(~isnan(sum(tmp,2)),:) );
            tmp1            = corr( [tmp(~isnan(sum(tmp,2)),1), log(tmp(~isnan(sum(tmp,2)),2))] );
            CUSTSZWT(t,i)   = tmp1(1,2); clear tmp*
        end
    end
    if mod(i,1000)==0, i, end
end

% Build matrices of supplier information (for customer i)
SUPPNUM     = nan(T,N);
SUPPOUT     = nan(T,N);
SUPPHERF    = nan(T,N);
SUPPSZWT    = nan(T,N);
for i=1:N
    loci            = find(linkages.cpermno==permlist(i));
    for t=1:T
        loct        = find(linkages.dates(loci)==yrlist(t));
        if isempty(loct), continue, end
        loc             = loci(loct);
        
        % This defines weights as fraction of a customers purchases coming
        % from each of its suppliers
        wt              = max(0,linkages.percsale(loc).*linkages.sale(loc)); wt = wt / nansum(wt);
        wt( wt <= 0 | isinf(wt)) = nan; if max(wt) > 1, wt = wt / nansum(wt); end

        wt2              = linkages.percsale(loc).*linkages.sale(loc) ./ linkages.ctotsale(loc); 
        wt2( wt2 <= 0 | isinf(wt2)) = nan; if max(wt2) > 1, wt2 = wt2 / nansum(wt2); end
        
        wt3              = linkages.percsale(loc).*linkages.sale(loc) ./ max(linkages.ctotsale(loc),nansum(linkages.percsale(loc).*linkages.sale(loc))); 
        wt3( wt3 <= 0 | isinf(wt3)) = nan; if max(wt3) > 1, wt3 = wt3 / nansum(wt3); end
        
        %wt = wt3;
        
        if nansum(wt) <=0, continue, end
        SUPPNUM(t,i)    = sum(~isnan(wt));
        SUPPOUT(t,i)    = nansum( wt );
        SUPPHERF(t,i)   = nansum( wt.^2 );
        
        tmp             = [linkages.percsale(loc).*linkages.sale(loc)./linkages.ctotsale(loc),linkages.sale(loc)];
        if sum(~isnan(sum(tmp,2)))>2
%             tmp1            = corr( tmp(~isnan(sum(tmp,2)),:) );
            tmp1            = corr( [tmp(~isnan(sum(tmp,2)),1), log(tmp(~isnan(sum(tmp,2)),2))] );
            SUPPSZWT(t,i)   = tmp1(1,2); clear tmp*
        end
    end
    if mod(i,1000)==0, i, end
end

SUPPHERF(SUPPHERF>1|SUPPHERF<=0)            = nan;
CUSTHERF(CUSTHERF>1|CUSTHERF<=0)            = nan;
SUPPNUM(SUPPNUM==0)                         = nan;
CUSTNUM(CUSTNUM==0)                         = nan;
SUPPOUT(SUPPOUT>1|SUPPOUT<=0)               = nan;
CUSTOUT(CUSTOUT>1|CUSTOUT<=0)               = nan;
%nansummary([vec(CUSTNUM) vec(SUPPNUM)])
%nansummary([vec(CUSTHERF) vec(SUPPHERF)])
%nansummary([vec(CUSTOUT) vec(SUPPOUT)])

LINKAGES.permlist      = permlist;
LINKAGES.yrlist        = yrlist;
LINKAGES.CUSTNUM       = CUSTNUM;
LINKAGES.CUSTOUT       = CUSTOUT;
LINKAGES.CUSTHERF      = CUSTHERF;
LINKAGES.CUSTSZWT      = CUSTSZWT;
LINKAGES.SUPPNUM       = SUPPNUM;
LINKAGES.SUPPOUT       = SUPPOUT;
LINKAGES.SUPPHERF      = SUPPHERF;
LINKAGES.SUPPSZWT      = SUPPSZWT;
save('../output/CleanData/LINKAGES 170301','LINKAGES')
%[yrlist, sum(~isnan(LINKAGES.SUPPNUM),2), sum(~isnan(LINKAGES.CUSTNUM),2)]

%mean(sum(~isnan(LINKAGES.SUPPNUM),2))
%mean(sum(~isnan(LINKAGES.CUSTNUM),2))
















