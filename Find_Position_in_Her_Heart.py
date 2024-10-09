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
    return -down_direction

#Dummy data
for i in range(1000):
    if i==0:
        g=np.array(account_for_g(0,-6.94,-6.94))
    rotation(np.array([np.pi/30,np.pi/3,np.pi/25]),i/100)
    motion(3,1,5,i/100)
    ta.append(i/100)
    tg.append(i/100)

ver_dir=calculate_lab_motion(g)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot(sx, sy, sz)
s_abs=np.sqrt(sx[-1]**2+sy[-1]**2+sz[-1]**2)
print(type(sy_lab[0]))

ax.quiver(*[0,0,0], *ver_dir, color='red', length=s_abs, arrow_length_ratio=0.1)
midpoint = s_abs*ver_dir / 2
ax.text(*midpoint, 'Upwards', color='red', fontsize=8, ha='center', va='bottom')

A, B, C, D = ver_dir[0], ver_dir[1], ver_dir[2], 0  # Plane equation: x + y + z - 1 = 0

# Create a grid for the plane
x_plane = np.linspace(-s_abs, s_abs, 2)
y_plane = np.linspace(-s_abs, s_abs, 2)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)
z_plane = (D - A * x_plane - B * y_plane) / C  # Solve for z
ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='cyan')

def update(num, x, y, z, line):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,
ani = FuncAnimation(fig, update, frames=len(ta), fargs=[sx, sy, sz, line], interval=1)
plt.show() 
