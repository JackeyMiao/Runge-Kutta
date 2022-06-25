clear
clc
%初始参数
param=[0;5000;2*340;-30/360*2*pi];
%依次步长0.01，0.1，1
[param_list_1,time_list_1]=calculate(0.01,1000,param);
[param_list_2,time_list_2]=calculate(0.5,1000,param);
[param_list_3,time_list_3]=calculate(1,1000,param);