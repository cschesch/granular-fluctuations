function [duplicates, single, loc_duplicates, loc_single] = findduplicates(id)

%id = [1 2 3 3 3 4 4 4 4 8]';
[~, locB]  = ismember(unique(id), id);
[locA, ~]     = ismember((1:length(id))',locB);
duplicates = unique(id(~locA)); duplicates(isnan(duplicates)) = [];
[locA, ~]     = ismember(id, duplicates);
single     = unique(id(~locA)); single(isnan(single)) = [];
loc_duplicates = ismember(id,duplicates);
loc_single     = ismember(id,single);


