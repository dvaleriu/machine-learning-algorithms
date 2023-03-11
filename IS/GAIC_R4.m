% GAIC_R3    Module that evaluates the optimum structural 
%            orders through Akaike criterion, generalized 
%            by Rissanen (GAIC_R), when identifying a 
%            model with 4 structural indexes 
%            (such as: Box-Jenkins). 
%
% Inputs:	Lambda # estimated noise variance matrix 
%               N      # size of measured data set
%
% Outputs:      nf     # optimum row order
%           	nb     # optimum column order 
%               nc     # optimum layer order
%               nd     # optimum cube
%
%           	GAICR  # values of GAIC_R criterion
%
% Explanation:  The 4D array of estimated noise variances 
%               includes Na rows, Nb columns and Nc layers. 
%               The GAIC_R criterion is evaluated for each 
%               element (i,j,k, l) of matrix. To select the optimum 
%               structural orders na, nb and nc, the smallest 
%               of these values is considered. Note that the 
%               structural indices result by decrementing 
%               nb, nc, nd and nf with 1. 
%
% Author:   Dan Stefanoiu (*)
% Revised:  Dan Stefanoiu (*)
%           Lavinius Ioan Gliga (*)
%
% Created: January  12, 2016
% Revised: November 24, 2020
%          August    9, 2018
%
% Copyright: (*) "Politehnica" Unversity of Bucharest, ROMANIA
%                Department of Automatic Control & Computer Science
%

function [nf,nb,nc,nd,GAICR] = GAIC_R4(Lambda,N) 

%
% BEGIN
% 
% Messages 
% ~~~~~~~~
	FN  = '<GAIC_R4>: ' ; 
	E1  = [FN 'Missing, empty or null N. Empty outputs. Exit.'] ; 
	E2  = [FN 'Missing or empty Lambda. Empty outputs. Exit.'] ;
%
% Faults preventing
% ~~~~~~~~~~~~~~~~~
nb = [] ;
nc = [] ; 
nd = [] ;
nf = [] ;
GAICR = [] ; 
if (nargin < 1)
   war_err(E2) ; 
   return ;
end
if (isempty(Lambda))
   war_err(E2) ; 
   return ;
end 
Lambda = abs(Lambda) ; 
Lambda(~Lambda) = eps ; 
if (nargin < 2)
   war_err(E1) ; 
   return ;
end 
if (isempty(N))
   war_err(E1) ; 
   return ;
end 
N = abs(round(N(1))) ; 
if (~N)
   war_err(E1) ; 
   return ;
end 
% 
% Evaluating the GAIC_R criterion
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[~, ~, ~,Nf] = size(Lambda) ; 
nb = zeros(1, Nf);
nc = zeros(1, Nf);
nd = zeros(1, Nf);
Gmin = zeros(1, Nf);
for nf=1:Nf
   [Nb,Nc,Nd,G] = GAIC_R3(Lambda(:,:,:,nf)*exp((nf-1)*log(N)/N),N) ; 
   nb(nf) = Nb ; 
   nc(nf) = Nc ;
   nd(nf) = Nd ;
   GAICR = cat(4,GAICR,G) ; 
   Gmin(nf) = G(Nb,Nc,Nd) ; 
end 
[~,nf] = min(Gmin) ; 
nb = nb(nf) ; 
nc = nc(nf) ; 
nd = nd(nf) ;

%
% END
%