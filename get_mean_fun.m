function [sample_mean, loss, travel_rute] = get_mean_fun(x, t, eps, learn_rate)
    [m, n] = size(x); % n is sample size, m is the dimension
    ini_p =  randn(m,1);
    ini_p = ini_p/norm(ini_p);
    travel_rute(:,1) = ini_p;
    cnt = 0;
    shrink_size = 100;
    while shrink_size> eps
        for i = 1:n
            v(:,i) = inv_exp2(t, ini_p, x(:,i));
        end 
        v_mean = learn_rate * mean(v,2);
        upd_p = exp2(t, ini_p, v_mean);
        shrink_size = acos(trapz(t, ini_p.*upd_p));
        for j = 1:n
            temp(j) = acos(trapz(t,ini_p.*x(:,i)));
        end
        cnt = cnt+1;
        loss(cnt)= sum(temp);
        ini_p = upd_p ; 
        travel_rute(:,cnt+1) = ini_p;
    end  
    sample_mean = ini_p;
end
