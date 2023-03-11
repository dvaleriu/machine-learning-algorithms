function Mid = armax_e(Did, orders)

if nargin < 1
    war_err(E1);
end
if isempty(Did)
    war_err(E1);
end
if nargin < 2
    war_err(E2);
    orders = [5 5 5 1];
end
if isempty(orders)
    war_err(E2);
    orders = [5 5 5 1];
end

%ordine ARMAX
na = orders(1);
nb = orders(2);
nc = orders(3);
nk = orders(4);

%ordine aproximare ARMAX prin ARX
nax = min([max([na, nb, nc]) , 10]);
nbx = nax;


%ARX pt estimare zgomot etapa1
arx_m = arx(Did, [nax nbx nk]);

%estimarea zgomotului 
e = pe(arx_m, Did);
e = e.y;

y = Did.y;
u = Did.u;
N = numel(e);

y = [zeros(na, 1); y];
e = [zeros(nc, 1); e];
u = [zeros(nb, 1); u];

R_N = zeros(na+nb+nc, na+nb+nc);
r_n = zeros(na+nb+nc, 1);

for i = 1:N
    phi = [];
    phi = [-y(i+na-1:-1:i); u(i+nb-1:-1:i); e(i+nc-1:-1:i)];
    R_N = R_N + 1/N *(phi * phi');
    r_n = r_n + 1/N * phi * Did.y(i);
end

theta = R_N\r_n;

%estimare parametri etapa 2
A = [1; theta(1:na)]';
B = [0; theta(na+1:na+nb)]';
C = [1; theta(na+nb+1:end)]';

Mid = idpoly(A, B, C, [], [], 1, 1);

%dispersia zgomotuluui
e = pe(Mid, Did);
Mid.NoiseVariance = norm(e.y)/sqrt(N-na-nc-nb);


end