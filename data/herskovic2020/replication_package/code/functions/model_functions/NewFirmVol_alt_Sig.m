function [out, ModelNewFirms] = NewFirmVol(Ntop,theta_guess,N,nloop,loc_exclude,FracNewFirms,sigtype)

[model_moments_guess_full_table, Sim_frac_hit_constraint, SZ_shr, CorrWS, Model_Output] = ...
                                    simulate_model_alt_sig(theta_guess(1:7),N,Ntop,nloop,loc_exclude,sigtype);

for s = 1:length(Model_Output.Sim_data_S(1,:))
    S       = Model_Output.Sim_data_S(:,s);
    sig2 = exp(Model_Output.Sim_data_sig(:,s));

    %locNewFirm     = cumsum(S/sum(S)) <= FracNewMktCap;
    locNewFirm     = S <= prctile(S,FracNewFirms*100);
        
    ModelNewFirms.FracNewFirms(s)  = numel(S(locNewFirm)) / numel(S);
    ModelNewFirms.FracNewMktCap(s) = sum(S(locNewFirm))   / sum(S);
    ModelNewFirms.NewFirmsVolRatio(s)   = nanmean(sig2(locNewFirm)) / nanmean(sig2(~locNewFirm)) - 1;
    ModelNewFirms.VolNewFirms(s)   = nanmean(sig2(locNewFirm));
    ModelNewFirms.VolOldFirms(s)   = nanmean(sig2(~locNewFirm));
end
ModelNewFirms.FracNewFirms  = mean(ModelNewFirms.FracNewFirms);
ModelNewFirms.FracNewMktCap = mean(ModelNewFirms.FracNewMktCap);
ModelNewFirms.NewFirmsVolRatio = mean(ModelNewFirms.NewFirmsVolRatio);
ModelNewFirms.VolNewFirms   = mean(ModelNewFirms.VolNewFirms);
ModelNewFirms.VolOldFirms   = mean(ModelNewFirms.VolOldFirms);
out   = ModelNewFirms.NewFirmsVolRatio;

disp([num2str(Ntop,' %.0f  ') '  ' num2str(out,' %.2f ')])
   