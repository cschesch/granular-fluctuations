function [pbase] = p_func(S,Z,nu,N,zetap)
    pbase = N^(1-zetap)*(S.^nu/sum(S.^nu))/Z; % S is an Nx1 column vector
    pbase = repmat(pbase,1,N); % pbase is an NxN matrix of probs
    pbase = max(min(pbase,1),0); % prob. between 0 and 1
end