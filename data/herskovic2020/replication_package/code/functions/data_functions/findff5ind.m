function ffind = findff5ind(sic)
%  1 Cnsmr  Consumer Durables, NonDurables, Wholesale, Retail, and Some Services (Laundries, Repair Shops)
if sic>=          0100   && sic<=  0999   , ffind = 1;
elseif sic>=          2000   && sic<=  2399   , ffind = 1;  
elseif sic>=          2700   && sic<=  2749   , ffind = 1;  
elseif sic>=          2770   && sic<=  2799   , ffind = 1;  
elseif sic>=          3100   && sic<=  3199   , ffind = 1;  
elseif sic>=          3940   && sic<=  3989   , ffind = 1;  
elseif sic>=          2500   && sic<=  2519   , ffind = 1;  
elseif sic>=          2590   && sic<=  2599   , ffind = 1;  
elseif sic>=          3630   && sic<=  3659   , ffind = 1;  
elseif sic>=          3710   && sic<=  3711   , ffind = 1;  
elseif sic>=          3714   && sic<=  3714   , ffind = 1;  
elseif sic>=          3716   && sic<=  3716   , ffind = 1;  
elseif sic>=          3750   && sic<=  3751   , ffind = 1;  
elseif sic>=          3792   && sic<=  3792   , ffind = 1;  
elseif sic>=          3900   && sic<=  3939   , ffind = 1;  
elseif sic>=          3990   && sic<=  3999   , ffind = 1;  
elseif sic>=          5000   && sic<=  5999   , ffind = 1;  
elseif sic>=          7200   && sic<=  7299   , ffind = 1;  
elseif sic>=          7600   && sic<=  7699   , ffind = 1;  

%  2 Manuf  Manufacturing, Energy, and Utilities
elseif sic>=          2520   && sic<=  2589   , ffind = 2;  
elseif sic>=          2600   && sic<=  2699   , ffind = 2;  
elseif sic>=          2750   && sic<=  2769   , ffind = 2;  
elseif sic>=          2800   && sic<=  2829   , ffind = 2;  
elseif sic>=          2840   && sic<=  2899   , ffind = 2;  
elseif sic>=          3000   && sic<=  3099   , ffind = 2;  
elseif sic>=          3200   && sic<=  3569   , ffind = 2;  
elseif sic>=          3580   && sic<=  3621   , ffind = 2;  
elseif sic>=          3623   && sic<=  3629   , ffind = 2;  
elseif sic>=          3700   && sic<=  3709   , ffind = 2;  
elseif sic>=          3712   && sic<=  3713   , ffind = 2;  
elseif sic>=          3715   && sic<=  3715   , ffind = 2;  
elseif sic>=          3717   && sic<=  3749   , ffind = 2;  
elseif sic>=          3752   && sic<=  3791   , ffind = 2;  
elseif sic>=          3793   && sic<=  3799   , ffind = 2;  
elseif sic>=          3860   && sic<=  3899   , ffind = 2;  
elseif sic>=          1200   && sic<=  1399   , ffind = 2;  
elseif sic>=          2900   && sic<=  2999   , ffind = 2;  
elseif sic>=          4900   && sic<=  4949   , ffind = 2;  

%  3 HiTec  Business Equipment, Telephone and Television Transmission
elseif sic>=          3570   && sic<=  3579   , ffind = 3;  
elseif sic>=          3622   && sic<=  3622   , ffind = 3;  
elseif sic>=          3660   && sic<=  3692   , ffind = 3;  
elseif sic>=          3694   && sic<=  3699   , ffind = 3;  
elseif sic>=          3810   && sic<=  3839   , ffind = 3;  
elseif sic>=          7370   && sic<=  7372   , ffind = 3;  
elseif sic>=          7373   && sic<=  7373   , ffind = 3;  
elseif sic>=          7374   && sic<=  7374   , ffind = 3;  
elseif sic>=          7375   && sic<=  7375   , ffind = 3;  
elseif sic>=          7376   && sic<=  7376   , ffind = 3;  
elseif sic>=          7377   && sic<=  7377   , ffind = 3;  
elseif sic>=          7378   && sic<=  7378   , ffind = 3;  
elseif sic>=          7379   && sic<=  7379   , ffind = 3;  
elseif sic>=          7391   && sic<=  7391   , ffind = 3;  
elseif sic>=          8730   && sic<=  8734   , ffind = 3;  
elseif sic>=          4800   && sic<=  4899   , ffind = 3;  

%  4 Hlth   Healthcare, Medical Equipment, and Drugs
elseif sic>=          2830   && sic<=  2839   , ffind = 4;  
elseif sic>=          3693   && sic<=  3693   , ffind = 4;  
elseif sic>=          3840   && sic<=  3859   , ffind = 4;  
elseif sic>=          8000   && sic<=  8099   , ffind = 4;  

%  5 Other  Other    && sic<=     && sic<=   Mines, Constr, BldMt, Trans,
%  Hotels, Bus Serv, Entertainment, Finance
else
ffind = 5;
end