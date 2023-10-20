function [prob, SSE_diff] = cal_joint_ratio_mulf(sigma1, q, g, t, q_new, q_old)
    [M, N] = size(q);
    SSE_new = 0;  
    SSE_old = 0; 
    for m = 1: M
        g_temp = g(m,:);
        q_m = q(m,:);
        temp_in1 = cumtrapz(t, EXP(t, g_temp).^2);
        temp_in1 = round(temp_in1/temp_in1(end)*(N-1))+1;

        SSE_new = SSE_new + (norm(q_new - q_m(temp_in1).*EXP(t,g_temp')))^2;  
        SSE_old = SSE_old + (norm(q_old - q_m(temp_in1).*EXP(t,g_temp')))^2; 
    end

    SSE_diff = SSE_new - SSE_old;
    prob = exp(-1/(2*sigma1^2)*SSE_diff);
end