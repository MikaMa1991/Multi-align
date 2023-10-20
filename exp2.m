function h = exp2(t,p,v)
    norm_v = sqrt(trapz(t, v.*v));
    q = cos(norm_v)*p + sin(norm_v)/norm_v*v;
    h = q/sqrt(trapz(t, q.^2));  %normalize
end
 
