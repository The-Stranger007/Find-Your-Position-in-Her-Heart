import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
ax_truely=[0]
vx=[0] #Velocity in x,y,z
vy=[0]
vz=[0]
sx=[0] #Displacement in x,y,z
sy=[0]
sz=[0]
rotation=[np.zeros((3,3))]
theta_x=[0]
theta_y=[0]
theta_z=[0]
t=[0]
#theta is the angular displacement from the respective axis in radians

def motion(ax,ay,az,wx,wy,wz,dta,dtg): #dta is the change in time from last accelerometer reading, dtg is that for gyroscope reading
    #in user frame: +x to the right of user, +y upwards, +z into the page

    theta_x.append(wx*dtg+theta_x[-1])
    theta_y.append(wy*dtg+theta_y[-1])
    theta_z.append(wz*dtg+theta_z[-1])

    r=R.from_rotvec(np.array([theta_x[-1],theta_y[-1],theta_z[-1]]))
    r_inv=r.inv()
    a_true=r_inv.apply(np.array([ax,ay,az]))
    ax_truely.append(a_true[0])
    

    vx.append(a_true[0]*dta+vx[-1]) #integrating acceleration to get velocity in user frame
    vy.append(a_true[1]*dta+vy[-1])
    vz.append(a_true[2]*dta+vz[-1])
    sx.append(vx[-2]*dta) #[-2] is used as we want to find change in s from the v at time for last data input. Since we already appended v once, the corresponding v will be v[-2] 
    sy.append(vy[-2]*dta)
    sz.append(vz[-2]*dta) #Find position vector with respect to position at t=0
    t.append(dta+t[-1])
    return
for i in range(10000):
    motion(1,0,0,np.pi/30,np.pi/30,np.pi/25,0.01,0.01)

#Testing:
t.remove(0)
theta_y.remove(0)
ax_truely.remove(0)
sx.remove(0)
sy.remove(0)
sz.remove(0)
s=[]
for i in range(len(sx)):
    s.append(np.sqrt(sx[i]**2+sy[i]**2+sz[i]**2))
plt.plot(t,sx)
plt.xlabel('time/s',fontsize=18)
plt.ylabel('Displacement in lab frame x direction / m',fontsize=18)
plt.show()

