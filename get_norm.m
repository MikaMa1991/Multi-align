function [norm_] = get_norm(N, q1, q2, gamma)
    time_gap = 1/(N-1);
    dot_r = gradient(gamma)/time_gap;
    temp_t=round(gamma/gamma(end)*(N-1))+1;
    temp = q1-sqrt(dot_r)*q2(temp_t);
    norm_ = norm(temp)^2;
end