function [param_list,time_list]=calculate(step,range,param)
    %step=1;
    %range=1000;
    %param=[0;5000;2*340;-30/360*2*pi];
    param_t=param;
    param_list=[];
    time_list=[];
    for t=0:step:range
        param_list=[param_list param_t];
        time_list=[time_list t];
        if(param_t(2)<0)
            break;
        end
        param_t=RungeKutta(param_t,step,@derive);
        %param=[param param_t];
    end

end
