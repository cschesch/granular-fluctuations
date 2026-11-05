function [out, ModelNewFirms] = SizeDispModel(Ntop,theta_guess,N,nloop,loc_exclude)

model_moments_guess_full_table = simulate_sizeOnly(theta_guess(1:7),N,Ntop,nloop,loc_exclude);

model_moments_NormX = nanmean((model_moments_guess_full_table),2);
out = model_moments_NormX(2);

disp([num2str(Ntop,' %.0f  ') '  ' num2str(out,' %.2f ')])
   