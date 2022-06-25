function param_f = RungeKutta(param,step,func)
    %构造变量序列
    %param_list=[param];
    %构造时间序列
    %T_list=[0];
    %构造迭代变量
    %param_t=param;
    %for t=0:step:range
    K1=func(param);
    K2=func(param+0.5*step*K1);
    K3=func(param+0.5*step*K2);
    K4=func(param+step*K3);
    param_f=param+step/6*(K1+2*K2+2*K3+K4);
    %param_list=[param_list param_t];
    %T_list=[T_list t];
    %判断是否落地
end