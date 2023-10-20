% Compute the SRVF of a given function.

function g_srvf = EXP(t, g)
    dist = sqrt(trapz(t, g.^2));
    h = cos(dist) + sin(dist)/dist*g;
    g_srvf = h/sqrt(trapz(t, h.^2));  %normalize
end