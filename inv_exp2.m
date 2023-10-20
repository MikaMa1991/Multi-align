function v = inv_exp2(t,p,q)
    theta = acos(trapz(t, p.*q));
    v = theta/sin(theta)*(q - p*cos(theta));
end
