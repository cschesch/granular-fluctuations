clear, addpath(genpath('functions\'))
%**************************************************************************
% Load
%**************************************************************************
yrlist  = (1926:2016)';
fid     = fopen('../data/all crsp monthly ret prc exchcd shrout sic 1926-2016.txt');
data    = textscan(fid,'%n%n%*n%n%n%n%n%n%n','headerlines',1,'delimiter','\t','TreatAsEmpty',{'A','B','C','P','T','S','Z'});

% PERMNO	date	SHRCD 	SICCD	DLRET	PRC	RET	SHROUT
permno  = data{1};
dates   = data{2};
exch    = data{3};
sic     = data{4};
dlret   = data{5};
prc     = abs(data{6});
ret     = data{7};
shrout  = 1000*abs(data{8}); % Reported in thousands
clear data fid

prc(prc==0)         = nan;
shrout(shrout==0)   = nan;
sic(sic==0)         = nan;
exch(exch==0)         = nan;
me                  = prc.*shrout;
permlist            = unique(permno);

% Check data
% nansummary(ret)
% nansummary(dlret)
% nansummary(dates)
% nansummary(permno)
% nansummary(prc)
% nansummary(sic)
% nansummary(shrout)

% Delist accounting
loc         = find(~isnan(ret) & ~isnan(dlret));                            
ret(loc)    = (1+ret(loc)).*(1+dlret(loc))-1;
loc         = find(isnan(ret) & ~isnan(dlret));
ret(loc)    = dlret(loc);
clear dlret* loc fid data

% Build ME
T       = length(yrlist);
N       = length(permlist);
yrmo    = floor(dates/100);
ME      = nan(T,N);
SIC     = nan(T,N);
EXCH    = nan(T,N);
for t=1:T
    loct    = find(yrmo==yrlist(t)*100+12);
    met     = me(loct);
    sict    = sic(loct);
    excht   = exch(loct);
    permt   = permno(loct);
    parfor i=1:N
        loci   = find(permt==permlist(i));
        if ~isempty(loci)
            ME(t,i)     = met(loci);
            SIC(t,i)    = sict(loci);
            EXCH(t,i)   = excht(loci);
        end
    end
    clear met permt sict
    t
end
CRSPMONTHLY.ME       = ME;
CRSPMONTHLY.SIC      = SIC;
CRSPMONTHLY.EXCH     = EXCH;
CRSPMONTHLY.permlist = permlist;
CRSPMONTHLY.yrlist   = yrlist;
save('../output/CleanData/CRSPMONTHLY July2017','CRSPMONTHLY')