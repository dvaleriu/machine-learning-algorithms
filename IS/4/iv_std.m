function Mid = iv_std(Did, na, nb, nk, N, Flag)

      switch Flag
             case '1' % pentru MCMMP
                 Mid = arx(Did,[na-1 nb-1 nk]) ;
             case '2' % pentru MVI - clasic
                 Mid = iv4(Did,[na nb nk]) ;
             case '3' % pentru MVI - partial filtrat
                 %se identifica modelul fltrului prin MCMMP
                 %se aplica partial filtrarea
                 model_aux = arx(Did, [na nb nk]);
                 uf = filter (model_aux.b, model_aux.a, Did.u);
                 uf = uf./sqrt(ones(N,1)*sum(uf.*uf)/N) ;
                 Did.u(1:na) = uf(1:na);
                Mid = iv(Did,[na, nb, nk], model_aux.a, model_aux.b);
                 case '4' % pentru MVI - partial filtrat
                 %se identifica modelul fltrului prin MCMMP
                 %se aplica partial filtrarea
                 model_aux = arx(Did, [na nb nk]);
                 uf = filter (model_aux.b, model_aux.a, Did.u);
                 uf = uf./sqrt(ones(N,1)*sum(uf.*uf)/N) ;
                 Did.u(na+1:na+nb) = uf(na+1:na+nb);
                 Mid = iv(Did,[na, nb, nk], model_aux.a, model_aux.b);
             case '5' % pentru MVI - total filtrat
                 %se aplica total filtrarea
                 model_aux = arx(Did, [na nb nk]);
                 uf = filter (model_aux.b, model_aux.a, Did.u);
                 uf = uf./sqrt(ones(N,1)*sum(uf.*uf)/N);
                 Did.u = uf;
                 Mid = iv(Did,[na nb nk],model_aux.a,model_aux.b) ;
           
         end


end



    
