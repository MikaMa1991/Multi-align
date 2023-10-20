function [prob, SSE_diff] = cal_joint_ratio(sigma1, q1, q2, t, g_new, g_old)
    N = length(t);
    temp_in1 = cumtrapz(t, EXP(t, g_new).^2);
    temp_in1 = round(temp_in1/temp_in1(end)*(N-1))+1;
    SSE_new = (norm(q1 - q2(temp_in1).*EXP(t,g_new')))^2;

    temp_in2 = cumtrapz(t, EXP(t, g_old).^2);
    temp_in2 = round(temp_in2/temp_in2(end)*(N-1))+1;
    SSE_old = (norm(q1 - q2(temp_in2).*EXP(t,g_old')))^2;
    SSE_diff = SSE_new - SSE_old;

    prob = exp(-1/(2*sigma1^2)*SSE_diff);
end