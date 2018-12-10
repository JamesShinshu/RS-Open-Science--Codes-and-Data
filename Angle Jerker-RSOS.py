import numpy as np
from matplotlib import pyplot as plot
from spm1d import rft1d



 ## Define function named 'trajectory'. 
 ## Imput: Moving position from xi to xf in time array t.
 ## Returns: Minimum jerk trajectory (From 'A Minimum-Jerk Trajectory' by Nevile Hogan (1984))
def trajectory(t,xi,xf):
    x = xi + (xf - xi)*(  10*( t/t.max(0) )**3 - 15*( t/t.max(0) )**4 + 6*( t/t.max(0) )**5   )
    return x
    
def GetANGLE(marker1, CoR, marker2, frame):
    tri_a = np.sum( ( (marker1[frame])-(CoR[frame]) )**2 )**0.5
    tri_b = np.sum( ( (marker2[frame])-(CoR[frame]) )**2 )**0.5
    tri_c = np.sum( ( (marker2[frame])-(marker1[frame]) )**2 )**0.5
    cos = (tri_a**2+tri_b**2-tri_c**2)/(2*tri_a*tri_b)
    ang = np.degrees(np.arccos(cos))
    return ang

def get_weights(Q, rate=4):
    x = np.linspace(0,10,Q)
    w0 = 1.0 / (1 + np.exp(5-rate*x) )
    w1 = w0[::-1]
    return (w0 + w1) - 1

# Create a random trajectory:
np.random.seed(1234567)
y0 = rft1d.randn1d(1,41,15)
w = get_weights(41, rate=8)
y = y0 * w

#t = range(40)
xi=161.35
xf=96.85

#t=DPhalanx[:,0]
t=np.linspace(0,40,41)      #Create an array of linear values from 0 to 2 (101 values)
x = trajectory(t,xi,xf)       #Create an array of minimum jerk trajectory
x_noise = rft1d.randn1d(1,41,5)
x_noise_clean = x_noise * w
amp = 2
xnew = x + amp * x_noise_clean

plot.close('all')                      #Close all plots
plot.figure(figsize=(8.2,6))
for i in range(12345,12350):
    
    #ax = plot.axes()
    #plot.setp(ax, xticks=[0, 0.1, 0.2, 0.3, 0.4], xticklabels=np.linspace(0.0,0.4,5))
    np.random.seed(i)
    y0 = rft1d.randn1d(1,41,15)
    w = get_weights(41, rate=8)
    y = y0 * w
    ax = plot.gca()
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    t=np.linspace(0,0.4,41)      #Create an array of linear values from 0 to 2 (101 values)
    x = trajectory(t,xi,xf)       #Create an array of minimum jerk trajectory
    x_noise = rft1d.randn1d(1,41,5)
    x_noise_clean = x_noise * w
    #AMP = np.linspace(0,10,21)
    amp = 2
    xnew = x + amp * x_noise_clean
    color = (i-12345)/10.0
    
###Text position
    xloc = 0.35
    yloc = 8
    ydloc = 160

#### (a) AMP = 0.5   ####  
    amp = 0.5
    clean_amp = amp * x_noise
    plot.subplot(2,2,1)
    plot.plot(t,clean_amp, color="%s" %color) #Plot array 't' and 'xnew'
    plot.ylim([-10,10])
    plot.xlim(0.0,0.4)
    plot.legend(loc=1)
    plot.xticks([0,0.1,0.2,0.3,0.4],np.linspace(0.0,0.4,5))
    plot.text(xloc,yloc,'(a)')
    plot.title('Gaussian noise SD = 0.5')
    plot.ylabel('Gaussian noise value [deg]')                ###Add labels on both X and Y axis

#### (c)    ####  
    size = 3
    sizeall = 0.5
    plot.subplot(2,2,3)
    plot.plot(t,xnew,linewidth =sizeall, color="%s" %color) #Plot array 't' and 'xnew'
    if i==12345:
        plot.plot(t,x, linewidth=size,c="black",label='Minimum jerk\ntrajectory')
        plot.plot(t,xnew,linewidth =sizeall, color="%s" %color,label='Noise\ntrajectory') #Plot array 't' and 'xnew'
    else:
        plot.plot(t,xnew,linewidth =sizeall, color="%s" %color) #Plot array 't' and 'xnew'
        plot.xlim(0.0,0.4)
        #plot.plot(t,clean_amp)
        plot.xticks([0,0.1,0.2,0.3,0.4],np.linspace(0.0,0.4,5))
        plot.legend(loc=3,prop={'size':10})
        plot.text(xloc,ydloc,'(c)')
        plot.xlabel('Time (s)')                    ###
    plot.ylabel('MP flexion angle [deg]')                ###Add labels on both X and Y axis     

#### (b) AMP = 3.5   ####        
    amp = 3.5
    clean_amp = amp * x_noise
    xnew = x + amp * x_noise_clean
      
    plot.subplot(2,2,2)
    plot.plot(t,clean_amp, color="%s" %color) #Plot array 't' and 'xnew'
    plot.ylim([-10,10])
    plot.xlim(0.0,0.4)
    plot.xticks([0,0.1,0.2,0.3,0.4],np.linspace(0.0,0.4,5))
    plot.legend(loc=1)
    plot.text(xloc,yloc,'(b)')
    plot.title('Gaussian noise SD = 3.5')
    

#### (d)    ####      
    plot.subplot(2,2,4)
    plot.plot(t,xnew,linewidth =sizeall, color="%s" %color) #Plot array 't' and 'xnew'
    if i==12345:
        plot.plot(t,x, linewidth=size,c="black")
        plot.plot(t,xnew,linewidth =sizeall, color="%s" %color) #Plot array 't' and 'xnew'
    else:
        plot.plot(t,xnew,linewidth =sizeall, color="%s" %color) #Plot array 't' and 'xnew'
        plot.xlim(0.0,0.4)
        plot.xticks([0,0.1,0.2,0.3,0.4],np.linspace(0.0,0.4,5))
        plot.legend(loc=3,prop={'size':10})
        plot.text(xloc,ydloc,'(d)')
        plot.xlabel('Time (s)')                    ###

plot.subplot(2,2,4)
plot.tight_layout()
plot.show()
