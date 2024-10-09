import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
v=[np.array([0,0,0])] #Velocity in x,y,z
s=[[0,0,0]] #Displacement in x,y,z
sx=[0]
sy=[0]
sz=[0]
a=[[0,0,0]]
w=[np.array([0,0,0])] #Angular velocity in x,y,z, from gyroscope
sy_lab=[]
sxz_lab=[]
theta=[[0,0,0]]
ta=[0] #t at each accelerometer reading
tg=[0] #t at each gyroscope reading
#theta is the angular displacement from the respective axis in radians

def account_for_g(ax0,ay0,az0): #Accounts for gravatational acceleration on Earth
    g_abs=np.sqrt(ax0**2+ay0**2+az0**2)
    if g_abs>=10.8 or g_abs<=8.8:
        print('Warning: Inaccurate accelerometer reading for g (',round(g_abs,2),')')
    return [ax0,ay0,az0]

def rotation(omega,t):
    #Updates rotation based on gyroscope reading
    w.append(omega)
    theta.append((theta[-1]+0.5*(w[-1]+w[-2])*(t-tg[-1]))%(2*np.pi)) #Integrates rotation vector from angular velocity (gyroscope reading)
     

def motion(ax,ay,az,t): 
    #Updates motion based on accelerometer reading
    #(t-ta[-1]) is the change in t from last accelerometer reading

    #Accounts for change in axis due to rotation
    r=R.from_rotvec(np.array(theta[-1]))
    r_inv=r.inv()
    a_true=r_inv.apply(np.array([ax,ay,az]))
    a_true-=g
    a.append(a_true)

    #Integrating acceleration and velocity along original axises, using trapezoidal rule
    v.append(0.5*(a[-1]+a[-2])*(t-ta[-1])+v[-1])
    s.append(0.5*(v[-1]+v[-2])*(t-ta[-1])+s[-1])
    sx.append(s[-1][0])
    sy.append(s[-1][1])
    sz.append(s[-1][2])
    
def calculate_lab_motion(g):
    down_direction=g/np.linalg.norm(g)
    for i in s:
        sy_lab.append(np.dot(i,down_direction))
        sxz_lab.append(np.linalg.norm(np.cross(i,down_direction)))

#Dummy data
for i in range(1000):
    if i==0:
        g=np.array(account_for_g(0,-6.94,-6.94))
    rotation(np.array([np.pi/30,np.pi/3,np.pi/25]),i/100)
    motion(3,1,5,i/100)
    ta.append(i/100)
    tg.append(i/100)

calculate_lab_motion(g)
# plt.subplot(2,2,1)
# plt.plot(ta,s)
# plt.xlabel('t/s',fontsize=18)
# plt.ylabel('Displacement in accelerometer axises at t=0/ m',fontsize=10)

# plt.subplot(2,2,2)
# plt.plot(ta,sy_lab)
# plt.xlabel('t/s',fontsize=18)
# plt.ylabel('Vertical Displacement/ m',fontsize=15)

# plt.subplot(2,2,3)
# plt.plot(ta,sxz_lab)
# plt.xlabel('t/s',fontsize=18)
# plt.ylabel('Horizontal Displacement/ m',fontsize=15)

# plt.subplot(2,2,4)
# rotvec=[]
# for i in theta:
#     if np.linalg.norm(i)==0:
#         rotvec.append([0,0,0])
#         continue
#     rotvec.append(i/np.linalg.norm(i))
# plt.plot(tg,rotvec)
# plt.xlabel('t/s',fontsize=18)
# plt.ylabel('Rotation Vector Elements',fontsize=15)
# plt.show()

# ax=plt.axes(projection='3d')
# ax.plot3D(sx,sy,sz)
# print(s[-1][0]-sx[-1])
# plt.show()

# fig=plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlim3d((0, np.max(sx) + np.ptp(sx)*0.05))
# ax.set_ylim3d((np.min(sy) - np.ptp(sy)*0.05, np.max(sy) + np.ptp(sy)*0.05))
# ax.set_zlim3d((np.min(sz) - np.ptp(sz)*0.05, np.max(sz) + np.ptp(sz)*0.05))
# line, = ax.plot([], [], lw=2)
# point, = ax.plot([], [], 'ro')

# def init():
#   line.set_data([], [])
#   point.set_data([], [])
#   return line, point

# def update(frame):
#   line.set_data(sx[:frame], sy[:frame], sz[:frame])
#   point.set_data(sx[frame], sy[frame],sz[:frame])
#   return line, point
# N=100
# # data = np.array(list(gen(N))).T
# ani = FuncAnimation(fig, update, N, frames=len(sx), interval=10000/N, blit=False)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot(sx, sy, sz)
def update(num, x, y, z, line):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,
ani = FuncAnimation(fig, update, frames=len(t), fargs=[x, y, z, line], interval=100)
plt.show()
