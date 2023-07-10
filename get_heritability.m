function [h2, se, p_wald, p_perm,v_w] = get_heritability(Y, X, K, se_flag, n_perm)
% Heritability analysis of multidimensional traits
% Reference: https://www.nature.com/articles/ncomms13291
%
% Input:
% Y: an Nsubj x Ndim matrix of multidimensional traits
% X: an Nsubj x Ncov matrix of covariates
% K: an Nsubj x Nsubj variance component matrix
% se_flag = 'precise': precise estimation of the se; se_flag = 'approx':
% approximate estimation of the se; 'precise' estimation can be very
% computationally expensive for moderate- and high-dimensional phenotypes;
% 'approx' is often accurate enough and is recommended
% n_perm: number of permutations; set Nperm = 0 if permutation inference is not needed
%
% Output:
% h2: heritability estimate
% se: standard error estimate
% p_wald: parametric Wald p-value
% p_perm: nonparametric permutation p-value; if Nperm = 0, PermPval = NaN
%%
[n_subj, n_cov] = size(X);
n_dim = size(Y,2);
%%
P0 = eye(n_subj) - X/(X'*X)*X';
[U,~,~] = svd(P0); U = U(:,1:n_subj-n_cov);
Y = U'*Y; K = U'*K*U;
n_subj = n_subj-n_cov;
%%
kappa = trace(K^2)/n_subj;
tau = trace(K)/n_subj;
vK = n_subj*(kappa-tau^2);

Qg = K-tau*eye(n_subj);
Qe = kappa*eye(n_subj)-tau*K;

Sg = Y'*Qg*Y/vK;
Se = Y'*Qe*Y/vK;
Sp = Sg+Se;
%%%%%%%%%%%%%
% M = trace(Sg)/(trace(Sg)+trace(Se));
% Mi = diag(Sg)./(diag(Sg)+diag(Se));
%%%%%%%%%%%%%
%Project Sg and Se on space of SDP matrices
Sg(isnan(Sg)) = 0;
[V_Sg,D_Sg]=eig(Sg);
D_Sg(D_Sg<0)=0;
Sg=V_Sg*D_Sg*inv(V_Sg);

Se(isnan(Se)) = 0;
[V_Se,D_Se]=eig(Se);
D_Se(D_Se<0)=0;
Se=V_Se*D_Se*inv(V_Se);

v_w=diag(Sg)./(diag(Sg)+diag(Se));

tg = trace(Sg);
te = trace(Se);
tp = tg+te;

% M = tg/tp;
h2 = max(min(tg/tp,1),0);
% h2 = tg/tp;
%%
if strcmp(se_flag, 'precise')
    df = [te, -tg]/tp^2;
    covt = zeros(2,2);
    for r = 1:n_dim
        for s = 1:n_dim
            Vrs = Sg(r,s)*K+Se(r,s)*eye(n_subj);
            covt = covt+2/vK^2*[trace(Qg*Vrs*Qg*Vrs), trace(Qg*Vrs*Qe*Vrs); trace(Qe*Vrs*Qg*Vrs), trace(Qe*Vrs*Qe*Vrs)];
        end
    end
    se = sqrt(df*covt*df');
elseif strcmp(se_flag, 'approx')
    se = sqrt(2*trace(Sp^2)/vK/tp^2);
end
%%
p_wald = 0.5-0.5*chi2cdf((h2/se)^2,1);
%%
if n_perm == 0
    p_perm = NaN;
    return
end

h2_perm = zeros(1,n_perm);
for s = 1:n_perm
    % disp(['----- Permutation-', num2str(s), ' -----'])
    
    if s == 1
        K_perm = K;
    else
        subj_perm = randperm(n_subj);
        K_perm = K(subj_perm,subj_perm);
    end
    
    Qg_perm = K_perm-tau*eye(n_subj);
    Qe_perm = kappa*eye(n_subj)-tau*K_perm;
    
    Sg_perm = Y'*Qg_perm*Y/vK;
    Se_perm = Y'*Qe_perm*Y/vK;
    
    tg_perm = trace(Sg_perm);
    te_perm = trace(Se_perm);
    
    h2_perm(s) = max(min(tg_perm/(tg_perm+te_perm),1),0);
end

p_perm = sum(h2_perm>=h2)/n_perm;
%%