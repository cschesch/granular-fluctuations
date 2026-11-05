function out = winsor(data,pct)

 

% data    :    data to be winsorized

% pct     :    percent of data (in total) to be replaced

 

low = prctile(data,pct/2);

hi  = prctile(data,100-pct/2);

 

for i=1:size(data,2)

    loc       = find(~isnan(data(:,i)));

    data(loc,i) = min(data(loc,i),hi(i));

    data(loc,i) = max(data(loc,i),low(i));

end

out = data;