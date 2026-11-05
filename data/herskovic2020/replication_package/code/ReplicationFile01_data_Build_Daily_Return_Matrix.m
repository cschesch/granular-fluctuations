clear, addpath(genpath('functions/'))
%******************************************************
% Read Data
%******************************************************
fid     = fopen('../data/CRSP shrcd 10-12 daily ret dlret 1926-2016.txt');

data = textscan(fid,'%n%n%*n%n%n','headerlines',1,'delimiter','\t','TreatAsEmpty',{'A','B','C','P','T','S'});

permno  = data{1};
dates   = data{2};
dlret   = data{3};
ret     = data{4};

% Delist accounting
loc         = find(~isnan(ret) & ~isnan(dlret));                            
ret(loc)    = (1+ret(loc)).*(1+dlret(loc))-1;
loc         = find(isnan(ret) & ~isnan(dlret));
ret(loc)    = dlret(loc);
clear dlret* loc fid data

daylist = unique(dates);
permlist= unique(permno);
T       = length(daylist);
N       = length(permlist);

%******************************************************
% Build Matrix
%******************************************************

RET = nan(T,N);
tic
for i=1:N
    loc         = find(permno==permlist(i));
    [~,ia,ib]   = intersect(daylist,dates(loc));
    RET(ia,i)   = ret(loc(ib));
    
    if floor(i/1000)*1000==i, i, toc, tic
        loc     = find(permno>permlist(i));
        permno  = permno(loc);
        dates   = dates(loc);
        ret     = ret(loc);
    end
end
%keep RET permlist daylist
save('../output/CleanData/Daily RET matrix CRSP shrcd 10-12 1926-2016','RET','permlist','daylist','-v7.3');
