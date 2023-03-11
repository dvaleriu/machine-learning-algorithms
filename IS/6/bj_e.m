function Mid = bj_e(Did, si)
if nargin < 1
    war_err(E1);
end
if isempty(Did)
    war_err(E1);
end
if nargin < 2
    war_err(E2);
    si = [5 5 5 5 1];
end
if isempty(si)
    war_err(E2);
    si = [5 5 5 5 1];
end

%ca la armax_e, se iau valorile indicilor structurali
nb = si(1);
nc = si(2);
nd = si(3);
nf = si(4);
nk = si(5);
armax_m = armax_e(Did, [nf+nd nb+nd nc+nf nk]);

%3 componente theta
AA = armax_m.a;
BB = armax_m.b;
CC = armax_m.c;
%3 radacini polinoame
rAA = roots(AA);
rBB = roots(BB);
rCC = roots(CC);
%radacinile comune ale polinoamelor a si c=>f, restul radacinilor sunt ale
%lui c
rF=intersect(rAA,rCC);
%pentru cazul complex, modulul radacinilor, iar apoi diferenta < eps
rF=abs(rF);
%MODULul pentru a sti cat de departe de cercul unitate
%nu stim oriebntarea, deci putem face si faza
%arctan(b/a),b-complex,a-real
rF = round(rF);
%puteam calcula si faza precum si arctan(b/a) 
for i=1:length(rF)-1
    for j=i+1:length(rF)-1
        if(abs(rF(i) - rF(j)) < 0.01)
            rF(i)=[];
        end
    end
end

F = poly(rF);
%restul radacinilor lui c
rC=setdiff(rCC,rF);
C = poly(rC);
%radacinile comune ale polinoamelor a si b=>d restul radacinilor sunt ale
%lui b
rD=intersect(rAA,rBB);
rD=abs(rD);
rD = round(rD);
for i=1:length(rD)-1
    for j=i+1:length(rD)-1
        if(abs(rD(i) - rD(j)) < 0.01)
            rD(i)=[];
        end
    end
end

D=poly(rD);

rB=setdiff(rBB,rD);
B = poly(rB);

Mid = idpoly([1],[zeros(1,nk) B],C,D,F,1) ; 
%dispersia
e = pe(Mid, Did);
N = numel(e.y);
Mid.NoiseVariance = norm(e.y)/sqrt(N-nd-nc-nb-nf);
%
% END
%

































































Mid1 = bj(Did,si);
Mid = idpoly([1],[Mid1.B],Mid1.C,Mid1.D,Mid1.F,1);
Mid.NoiseVariance=Mid1.NoiseVariance;

%
end