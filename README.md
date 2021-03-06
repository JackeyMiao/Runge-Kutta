# Runge-Kutta

## 算法描述

共包含Python与Matlab两种实现方式。

### Python

使用**四阶龙格库塔**计算空地导弹在铅垂平面内无动力滑行的质点弹道，画出了弹道曲线及速度随时间变化的曲线。同时，给出了不同计算步长对仿真结果的影响，并利用**拉格朗日插值**给出了落地时x坐标的精确值。

同时实现了定步长与**变步长**。

### Matlab

仅实现了定步长求解。

## 模型及初始条件

### 动力学模型

建立二维铅垂平面内的导弹质点运动学模型如下：
$$
\dot{x}=Vcos\theta\\
\dot{y}=Vsin\theta\\
\dot{V}=-\frac{D}{m}-gsin\theta\\
\dot{\theta}=\frac{gcos\theta}{V}+\frac{L}{mV}
$$
其中，
$$
D=\frac{1}{2}\rho V^2sC_x\\
L=\frac{1}{2}\rho V^2sC_y
$$
$x$为水平位移，$y$为竖直位移，$V$为导弹速度，$\theta$为弹道倾角，$D$为阻力，$m$为导弹质量，$L$为升力，$g$为重力加速度，$s$为参考面积，$\rho$为密度公式，$C_x$与$C_y$分别为阻力系数和升力系数。

**阻力系数：**
$$
C_x=
\left[\begin{matrix}
M^2 & M & 1
\end{matrix}\right]
·
\left[\begin{matrix}
0.0002 & 0.0038 & 0.1582\\
-0.0022 & -0.0132 & -0.8520\\
0.0115 & -0.0044 & 1.9712
\end{matrix}\right]
·
\left[\begin{matrix}
\alpha^2 & \alpha & 1
\end{matrix}\right]^T
$$
**升力系数：**
$$
C_y=
\left[\begin{matrix}
M^2 & M & 1
\end{matrix}\right]
·
\left[\begin{matrix}
-0.026 & 0.0651 & 0.4913
\end{matrix}\right]^T
·
\alpha
$$
**密度公式：**（h为高度，单位为米）
$$
\rho=1.225e^{-0.00015h}
$$
阻力系数和升力系数中攻角$α$的单位为度；假定飞行过程中攻角$α$始终为2度，仿真至导弹掉地为止。

### 四阶龙格-库塔法弹道积分算法

对于微分方程$\frac{dy}{dt}=f(y,t)$，$y_{m+1}$与$y_m$的关系由四阶龙格库塔算法可得：
$$
y_{m+1}=y_m+\frac{h}{6}(k_1+2k_2+2k_3+k_4)\\
k_1=f(t_m,y_m)\\
k_2=f(t_m+\frac{h}{2},y_m+\frac{h}{2}k_1)\\
k_3=f(t_m+\frac{h}{2},y_m+\frac{h}{2}k_2)\\
k_4=f(t_m+h,y_m+hk_3)
$$

### 初始条件

$m_0=260kg,s=0.24m^2,M_0=2,x_0=0,h_0=5000m,\theta_0=-30°,a=340m/s$
