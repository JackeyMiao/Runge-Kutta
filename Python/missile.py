'''
Description: 
Author: Jackeeee_M
Date: 2022-03-30 08:00:24
LastEditors: Jackeeee_M
LastEditTime: 2022-04-05 13:57:47
'''

from ctypes.wintypes import SIZE
from platform import java_ver
import numpy as np
import math
import matplotlib.pyplot as plt 

# 定义变量
m=260
s=0.24
M=2

x=[0]
y=[5000]
theta=[-30/360*2*math.pi]
v=[680]
time_f=[0]

x_result=[]
y_result=[]
theta_result=[]
v_result=[]

x_half=[]
y_half=[]
theta_half=[]
v_half=[]

error_x=[0]
error_y=[0]

alpha=2
h=0.1# 步长在这里
g=9.8
angle=[]
v1=[]# 加速度
time=[]
cnt=0



# 初始化
def init(h_new,x_new,y_new,theta_new,v_new):
    '''
    description:用于初始化初始值
    '''    
    global x,y,theta,v,h,cnt
    h=h_new
    x=[x_new]
    y=[y_new]
    theta=[theta_new]
    v=[v_new]
    cnt=0

# 定义x导的微分方程
def x_deriv(v_f,theta_f):
    x1=v_f*math.cos(theta_f)
    return x1
# 定义v导的微分方程
def y_deriv(v_f,theta_f):
    y1=v_f*math.sin(theta_f)
    return y1
# 定义V导的微分方程
def v_deriv(v_f,theta_f,y_f):
    global alpha,g,m
    a=np.array([[(v_f/340)**2,v_f/340,1]])
    b=np.array([[0.0002,0.0038,0.1582],
                [-0.0022,-0.0132,-0.8520],
                [0.0115,-0.0044,1.9712]])
    c=np.array([[alpha**2],
                [alpha],
                [1]])
    Cx=np.matmul(np.matmul(a,b),c)
    ro=1.225*math.e**(-0.00015*y_f)
    D=0.5*ro*v_f**2*s*Cx[-1][-1]
    v1=-D/m-g*math.sin(theta_f)
    return v1
# 定义θ导的微分方程
def theta_deriv(v_f,theta_f,y_f):
    global alpha,g,m
    a=np.array([[(v_f/340)**2,v_f/340,1]])
    b=np.array([[-0.026],
                [0.0651],
                [0.4913]])
    Cy=np.matmul(a,b)*alpha
    ro=1.225*math.e**(-0.00015*y_f)
    L=0.5*ro*v_f**2*s*Cy[-1][-1]
    theta1=-g*math.cos(theta_f)/v_f+L/(m*v_f)
    return theta1
# 定义全部导数
def param_deriv(v_f,theta_f,y_f):
    '''
    description:同时对x,y,v,theta进行求导
    '''    
    x1=x_deriv(v_f,theta_f)
    y1=y_deriv(v_f,theta_f)
    v1=v_deriv(v_f,theta_f,y_f)
    theta1=theta_deriv(v_f,theta_f,y_f)
    return x1,y1,v1,theta1

# 定义Runge-Kutta方程
def iter(step,x0,y0,v0,theta0,kx1,ky1,kv1,ktheta1):
    '''
    description:龙格库塔的K的一次更新
    '''    
    x1=x0+0.5*step*kx1
    y1=y0+0.5*step*ky1
    v1=v0+0.5*step*kv1
    theta1=theta0+0.5*step*ktheta1
    return x1,y1,v1,theta1

def RungeKutta(step):
    '''
    description:龙格库塔的一次更新
    ''' 
    v0,x0,y0,theta0=v[-1],x[-1],y[-1],theta[-1]
    kx1,ky1,kv1,ktheta1=param_deriv(v0,theta0,y0)
    x1,y1,v1,theta1=iter(step,x0,y0,v0,theta0,kx1,ky1,kv1,ktheta1)
    kx2,ky2,kv2,ktheta2=param_deriv(v1,theta1,y1)
    x2,y2,v2,theta2=iter(step,x0,y0,v0,theta0,kx2,ky2,kv2,ktheta2)
    kx3,ky3,kv3,ktheta3=param_deriv(v2,theta2,y2)
    x3,y3,v3,theta3=iter(step,x0,y0,v0,theta0,2*kx3,2*ky3,2*kv3,2*ktheta3)
    kx4,ky4,kv4,ktheta4=param_deriv(v3,theta3,y3)

    v_t=v0+step/6*(kv1+2*kv2+2*kv3+kv4)
    theta_t=theta0+step/6*(ktheta1+2*ktheta2+2*ktheta3+ktheta4)
    x_t=x0+step/6*(kx1+2*kx2+2*kx3+kx4)
    y_t=y0+step/6*(ky1+2*ky2+2*ky3+ky4)

    return v_t,theta_t,x_t,y_t

# 绘图
def draw():
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画图
    f, ax = plt.subplots(2, 2)
    # 设置主标题
    # f.suptitle('步长为'+str(h)+'时')


    # 误差图
    plt.subplot(2, 4, 1)
    # plt.plot(time, error_x,markersize=1,color=(0.5,0.,0.))
    # plt.plot(time, error_y,markersize=1,color=(0.,0.,0.5))
    plt.scatter(time[1:],error_x,marker='.', linewidths=0.01,color=(0.5,0.,0.),label='error_x')
    plt.scatter(time[1:],error_y,marker='.', linewidths=0.01,color=(0.,0.,0.5),label='error_y')
    plt.title("误差",fontsize=20)
    plt.xlabel("时间/s",fontsize=20)
    plt.ylabel("误差/m",fontsize=20)
    plt.ylim(ymin=0)
    plt.xlim(xmax=80,xmin=0)
    plt.legend(loc=(0.5,0.),ncol=1,fontsize=14)
    


    # 速度曲线
    plt.subplot(2, 4, 2)
    plt.plot(time, v_result,markersize=1,color=(0.5,0.,0.))
    plt.title("导弹速度曲线",fontsize=20)
    plt.xlabel("时间/s",fontsize=20)
    plt.ylabel("速度/(m/s)",fontsize=20)
    plt.xlim(xmax=80,xmin=0)
    plt.ylim(ymax=1000,ymin=0)


    # 角度曲线
    plt.subplot(2, 4, 5)
    for i in range(0,len(theta_result)):
        angle.append(theta_result[i]/(2*math.pi)*360)
    plt.plot(time,angle,markersize=1,color=(0.5,0.,0.))
    plt.title("θ角度曲线",fontsize=20)
    plt.xlabel("时间/s",fontsize=20)
    plt.ylabel("θ/°",fontsize=20)
    plt.xlim(xmax=80,xmin=0)



    # 加速度曲线
    plt.subplot(2, 4, 6)
    for i in range(0,len(theta_result)):
        v1.append(v_deriv(v_result[i],theta_result[i],y_result[i]))
    plt.plot(time, v1,markersize=1,color=(0.5,0.,0.))
    plt.title("加速度曲线",fontsize=20)
    plt.xlabel("时间/s",fontsize=20)
    plt.ylabel("加速度a/(m/s2)",fontsize=20)
    plt.xlim(xmax=80,xmin=0)

    # 轨迹曲线
    plt.subplot(1,2,2)
    plt.plot(x_result,y_result,markersize=1,color=(0.5,0.,0.))
    plt.title("导弹弹道轨迹",fontsize=20)
    plt.xlabel("距离/m",fontsize=20)
    plt.ylabel("高度/m",fontsize=20)
    plt.xlim(xmax=10000,xmin=0)
    plt.ylim(ymax=7000,ymin=0)

    
    plt.show()

# 计算弹道
def calculate(step,x_new,y_new,theta_new,v_new):
    '''
    description: 判断落地为y是否大于0,落地后跳出循环，一次循环即一次迭代。
                 先同时做一次龙格库塔,然后依次更新。
                 用于定步长计算。
    '''    
    global x,y,v,theta,M,h,cnt
    # 一倍步长情况下
    init(step,x_new,y_new,theta_new,v_new)
    while(y[-1]>0):
        v_temp,theta_temp,x_temp,y_temp=RungeKutta(step)
        x.append(x_temp)
        y.append(y_temp)
        v.append(v_temp)
        theta.append(theta_temp)
        M=v[-1]/340
        # print(x[-1],y[-1])# 打印x距离
        cnt+=1
    print('用时：'+str(round(cnt*step*10)/10)+'s')

#计算误差
def calculate_error():
    '''
    description:用于定步长龙格库塔计算误差
    '''    
    print(len(x_result),len(x_half))
    for i in range(1,len(x_result)-1):
        error_x.append(abs(x_result[i]-x_half[2*i])*16/15)
        error_y.append(abs(y_result[i]-y_half[2*i])*16/15)

# 定步长龙格库塔
def RungeKutta_fixed(step,x_new,y_new,theta_new,v_new):
    global x_result,y_result,v_result,theta_result,x_half,y_half,v_half,theta_half,h
    calculate(step,x_new,y_new,theta_new,v_new)
    # 记录一倍步长
    x_result=x
    y_result=y
    theta_result=theta
    v_result=v
    # 生成时间序列
    for i in range(0,len(x)):
        time.append(h*i)
    # 记录半步长
    calculate(0.5*step,x_new,y_new,theta_new,v_new)
    x_half=x
    y_half=y
    theta_half=theta
    v_half=v
    calculate_error()
    print(len(x_result),len(x_half))# 迭代次数
    print(x[-2])
    h=2*h

# 变步长龙格库塔
def RungeKutta_float(step,threshold,x_new,y_new,theta_new,v_new):
    '''
    description:变步长的龙格库塔,其实step可以不用,因为是变步长 
    param {*step:步长
           *threshold:设置的阈值
           *x_new:x初值
           *y_new:y初值
           *theta_new:θ初值
           *v_new:v初值
    }
    '''    
    global x,y,v,theta,M,h
    global x_result,y_result,v_result,theta_result,time
    # 一倍步长情况下
    init(step,x_new,y_new,theta_new,v_new)
    while(y[-1]>0):
        v_temp1,theta_temp1,x_temp1,y_temp1=RungeKutta(step)
        v_temp2,theta_temp2,x_temp2,y_temp2=RungeKutta(step/2)
        # 变步长
        if abs(x_temp1-x_temp2)>=threshold or abs(y_temp1-y_temp2)>=threshold:
            while(1):
                step=step/2
                v_temp1,theta_temp1,x_temp1,y_temp1=RungeKutta(step)
                v_temp2,theta_temp2,x_temp2,y_temp2=RungeKutta(step/2)
                if abs(x_temp1-x_temp2)<=threshold and abs(y_temp1-y_temp2)<=threshold:
                    break
        else:
            while(1):
                step=step*2
                v_temp1,theta_temp1,x_temp1,y_temp1=RungeKutta(step)
                v_temp2,theta_temp2,x_temp2,y_temp2=RungeKutta(step/2)
                if abs(x_temp1-x_temp2)>=threshold or abs(y_temp1-y_temp2)>=threshold:
                    step=step/2
                    break
        # 分别记录一倍步长与半步长数据
        v_temp1,theta_temp1,x_temp1,y_temp1=RungeKutta(step)
        v_temp2,theta_temp2,x_temp2,y_temp2=RungeKutta(step/2)
        # 计算并更新结果及误差
        x.append(x_temp1)
        y.append(y_temp1)
        v.append(v_temp1)
        theta.append(theta_temp1)
        error_x.append(abs(x_temp1-x_temp2))
        error_y.append(abs(y_temp1-y_temp2))
        time_f.append(time_f[-1]+step)
        M=v[-1]/340
        print(x[-1],y[-1])# 打印x距离
    # 细微调整
    x_result=[0]
    x_result.extend(x)
    y_result=[5000]
    y_result.extend(y)
    theta_result=[-30/360*2*math.pi]
    theta_result.extend(theta)
    v_result=[680]
    v_result.extend(v)
    time=time_f
    time.append(time[-1])


# 拉格朗日插值
def Lagrange(x,y,expected):
    '''
    description:拉格朗日插值从而计算出最终的落地精确点,由于需要做到,求y=0时的x坐标,故在输入的时候可以xy坐标调换位置
    param {*x:拉格朗日的自变量x
           *y:拉格朗日的函数值y
           *expected:期望的x
    }
    return {*}
    '''    
    ans=0.0
    for i in range(len(y)):
        t=y[i]
        for j in range(len(y)):
            if i !=j:
                t*=(expected-x[j])/(x[i]-x[j])
        ans +=t
    return ans

# 变θ初始值的画图
def theta_change(gap,iter,step,x_new,y_new,theta_new,v_new):
    '''
    description:对θ的初始值进行更改,并根据结果绘图 
    param {*gap:每次θ更改的步长大小
           *iter:θ的步进次数
           *step:龙格库塔步长
           *theta_new:θ的初始值,之后以此为基准步进更新
    }
    '''    
    x_L=[]
    theta_L=[]
    for i in range(0,iter):
        theta_L.append(theta_new/(2*math.pi)*360)
        calculate(step,x_new,y_new,theta_new,v_new)
        x_result=x
        y_result=y
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(x_result,y_result,markersize=1,label=str(round(theta_new/math.pi/2*360))+'°')
        plt.title("变θ初始值导弹弹道轨迹",fontsize=20)
        plt.xlabel("距离/m",fontsize=20)
        plt.ylabel("高度/m",fontsize=20)
        theta_new+=gap
        plt.xlim(xmax=10000,xmin=0)
        plt.ylim(ymax=8000,ymin=0)
        x_L.append(Lagrange(y_result[-6:],x_result[-6:],0))

    plt.legend(loc=(0.,0.),ncol=1,fontsize=14)
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(theta_L,x_L,markersize=1)
    plt.title("随着θ初始值变化的落点变化曲线",fontsize=20)
    plt.xlabel("θ初始值/°",fontsize=20)
    plt.ylabel("距离/m",fontsize=20)
    plt.xlim(xmax=0,xmin=-90)
    plt.show()


# 变v初始值的画图
def v_change(gap,iter,step,x_new,y_new,theta_new,v_new):
    '''
    description:对v的初始值进行更改,并根据结果绘图 
    param {*gap:每次v更改的步长大小
           *iter:v的步进次数
           *step:龙格库塔步长
           *v_new:v的初始值,之后以此为基准步进更新
    }
    '''
    x_L=[]
    v_L=[]
    for i in range(0,iter):
        v_L.append(v_new)
        calculate(step,x_new,y_new,theta_new,v_new)
        x_result=x
        y_result=y
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(x_result,y_result,markersize=1,label=str(v_new)+'m/s')
        plt.title("变v初始值导弹弹道轨迹",fontsize=20)
        plt.xlabel("距离/m",fontsize=20)
        plt.ylabel("高度/m",fontsize=20)
        v_new+=gap
        plt.xlim(xmax=11000,xmin=0)
        plt.ylim(ymax=7000,ymin=0)
        x_L.append(Lagrange(y_result[-6:],x_result[-6:],0))
    
    plt.legend(loc=(0.,0.),ncol=1,fontsize=14)
    plt.subplot(1, 2, 2)
    plt.plot(v_L,x_L,markersize=1)
    plt.title("随着v初始值变化的落点变化曲线",fontsize=20)
    plt.xlabel("v初始值/(m/s)",fontsize=20)
    plt.ylabel("距离/m",fontsize=20)
    plt.xlim(xmax=1000,xmin=100)
    plt.show()

# 变m初始值的画图
def m_change(gap,iter,step,x_new,y_new,theta_new,v_new,m_new):
    '''
    description:对m的初始值进行更改,并根据结果绘图 
    param {*gap:每次m更改的步长大小
           *iter:m的步进次数
           *step:龙格库塔步长
           *m_new:m的初始值,之后以此为基准步进更新
    }
    ''' 
    x_L=[]
    m_L=[]
    global m   
    m=m_new
    for i in range(0,iter):
        m_L.append(m)
        calculate(step,x_new,y_new,theta_new,v_new)
        x_result=x
        y_result=y
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(x_result,y_result,markersize=1,label=str(m)+'kg')
        plt.title("变m初始值导弹弹道轨迹",fontsize=20)
        plt.xlabel("距离/m",fontsize=20)
        plt.ylabel("高度/m",fontsize=20)
        m+=gap

        plt.xlim(xmax=12000,xmin=0)
        plt.ylim(ymax=6000,ymin=0)
        x_L.append(Lagrange(y_result[-6:],x_result[-6:],0))
    
    plt.legend(loc=(0.,0.),ncol=1,fontsize=14)
    plt.subplot(1, 2, 2)
    plt.plot(m_L,x_L,markersize=1)
    plt.title("随着m变化的落点变化曲线",fontsize=20)
    plt.xlabel("质量/kg",fontsize=20)
    plt.ylabel("距离/m",fontsize=20)
    plt.xlim(xmax=1000,xmin=100)
    plt.show()

# 变y初始值的画图
def y_change(gap,iter,step,x_new,y_new,theta_new,v_new):
    '''
    description:对y的初始值进行更改,并根据结果绘图 
    param {*gap:每次y更改的步长大小
           *iter:y的步进次数
           *step:龙格库塔步长
           *y_new:y的初始值,之后以此为基准步进更新
    }
    '''
    x_L=[]
    y_L=[]
    for i in range(0,iter):
        y_L.append(y_new)
        calculate(step,x_new,y_new,theta_new,v_new)
        x_result=x
        y_result=y
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(x_result,y_result,markersize=1,label=str(y_new)+'m')
        plt.title("变y初始值导弹弹道轨迹",fontsize=20)
        plt.xlabel("距离/m",fontsize=20)
        plt.ylabel("高度/m",fontsize=20)
        y_new+=gap

        plt.xlim(xmax=11000,xmin=0)
        plt.ylim(ymax=8000,ymin=0)
        x_L.append(Lagrange(y_result[-6:],x_result[-6:],0))
    
    plt.legend(loc=(0.,0.),ncol=1,fontsize=14)
    plt.subplot(1, 2, 2)
    plt.plot(y_L,x_L,markersize=1)
    plt.title("随着y初始值变化的落点变化曲线",fontsize=20)
    plt.xlabel("y初始值/m",fontsize=20)
    plt.ylabel("距离/m",fontsize=20)
    plt.xlim(xmax=7500,xmin=3000)
    plt.show()


# 主函数
RungeKutta_fixed(0.1,0,5000,-30/360*2*math.pi,680)
# RungeKutta_float(1,100,0,5000,-30/360*2*math.pi,680)
draw()
# print('插值结果为'+str(Lagrange(y_result[-6:],x_result[-6:],0)))
# theta_change(10/360*2*math.pi,10,0.1,0,5000,-90/360*2*math.pi,680)
# v_change(100,10,0.1,0,5000,-30/360*2*math.pi,100)
# m_change(100,10,0.1,0,5000,-30/360*2*math.pi,500,100)
# y_change(500,10,0.1,0,3000,-30/360*2*math.pi,500)