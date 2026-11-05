clear, addpath(genpath('functions\'))

%**************************************************************************
% Load
%**************************************************************************
yrlist  = (1926:2016)';
fid     = fopen('../data/Compustat Qtrly Sales and GIC July2017.txt');
% GVKEY	LPERMNO	datadate	fyearq	fqtr	saleq	gic
data    = textscan(fid,'%n%n%n%*n%*n%*s%*s%*s%*s%*s%*s%*s%n%*s%n','headerlines',1,'delimiter','\t'); %492,225 observations
gvkey   = data{1};
permno  = data{2};
dates   = data{3};
sales   = 1000000*data{4}; % Reported in millions
gic     = data{5};
clear data fid

sales(sales<=0) = nan;
permlist        = unique(permno);

% Build quarterly YoY growth rates
gvlist  = unique(gvkey);
salesgr = nan(size(sales));
dates   = floor(dates/100);
for i=1:length(gvlist)
    loc0              = find(gvkey==gvlist(i));
    for i_permno = unique(permno(loc0))' % some gvkey have multiple permno
        loc = loc0(permno(loc0) == i_permno);
        [~,ia,ib]        = intersect(dates(loc),dates(loc)-100); 
        salesgr(loc(ib)) = log(sales(loc(ib))./sales(loc(ia)));

        if length(dates(loc)) ~= length(unique(dates(loc))) % some gvkey-permno have multiple dates (?)
            [~,loc_extra] = ismember(unique(dates(loc)),dates(loc));
            [loc_extra,~]=ismember((1:length(dates(loc)))',loc_extra);
            loc_extra = find(~loc_extra);
            loc_multi =find(ismember(dates(loc),dates(loc(loc_extra))));
            for i_dates = unique(dates(loc(loc_multi)))'
                loc_multi2 = loc_multi(dates(loc(loc_multi)) == i_dates);
                if max(sales(loc(loc_multi2))) ==  min(sales(loc(loc_multi2)))
                    sales(loc(loc_multi2)) = max(sales(loc(loc_multi2)));
                else
                    sales(loc(loc_multi2)) = nanmean(sales(loc(loc_multi2))); % nan
                end
                    salesgr(loc(ib)) = log(sales(loc(ib))./sales(loc(ia)));
                    %salesgr(loc(loc_multi2)) = salesgr(loc(loc_multi2(1)));
                    sales(loc(loc_multi2(2:end)))  = nan; 
            end
        end
    end
    if mod(i,1000)==0, i, end
end
salesgr = winsor(salesgr,2);

% Convert to calendar-year annual obs
T           = length(yrlist);
N           = length(permlist);
yr          = floor(dates/100);
SALES       = nan(T,N);
SALESVOL1Y  = nan(T,N);
SALESVOL5Y  = nan(T,N);
GIC         = nan(T,N);
for i=1:length(permlist)
    loc = find(permno==permlist(i));
    
    if length(loc)<4, continue, end
    yrtmp       = yr(loc);
    yrtmplist   = unique(yrtmp);
    
    for t=1:length(yrtmplist)
        loct = find(yrtmp==yrtmplist(t));
        
        if length(loct)<4, continue, end
        loct1           = loc(loct); 
        ix              = find(yrlist==yrtmplist(t));
        SALES(ix,i)     = sum(sales(loct1));
        GIC(ix,i)       = mode(gic(loct1));
        SALESVOL1Y(ix,i)= sqrt(4)*std(salesgr(loct1));
        
        loct = find(yrtmp>=yrtmplist(t) & yrtmp<yrtmplist(t)+5);
        if length(loct)<12, continue, end
        loct1           = loc(loct);    
        SALESVOL5Y(ix,i)= sqrt(4)*std(salesgr(loct1));
    end
    
    if floor(i/1000)*1000==i,i,end
end
SALESVOL1Y(SALESVOL1Y==0) = nan;
SALESVOL5Y(SALESVOL5Y==0) = nan;
COMPUSTAT.permlist  = permlist;
COMPUSTAT.yrlist    = yrlist;
COMPUSTAT.SALES     = SALES;
COMPUSTAT.GIC       = GIC;
COMPUSTAT.SALESVOL1Y= SALESVOL1Y;
COMPUSTAT.SALESVOL5Y= SALESVOL5Y;
save('../output/CleanData/COMPUSTAT July2017','COMPUSTAT')