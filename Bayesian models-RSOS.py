import numpy as np
from matplotlib import pyplot

def fA(x, a0, a1):
    '''
    Model A: Linear increase in preference with smoothness
    
    x: Jerk magnitude
    
    a0: Intercept (preference when jerk is hypothetically zero)
    
    a1: Slope
    '''
    return a0 + a1*x

def fB(x, b0, b1, b2):
    '''
    Model B" Linear increase in preference with smoothness, with flattened preference past a certain smoothness
    
    x: Jerk magnitude
    
    b0: Intercept (preference when jerk is hypothetically zero)
    
    b1: Slope
    
    b2: Jerk value at which preference flattens out
    '''
    f = b0 + b1*x
    f[x<b2] = b0 + b1*b2
    return f

def fC(x, c0, c1, c2):
    '''
    Model C" Linear increase in preference with smoothness, with reverse p reference past a certain smoothness
    
    x: Jerk magnitude
    
    c0: Intercept (preference when jerk is hypothetically zero)
    
    c1: Slope
    
    c2: Jerk value at which preference reverses
    '''
    f = c0 + c1*x
    f[x<c2] = (c0 + 2.0*c1*c2) -c1*x[x<c2]
    return f

fnameCSV = '_directory_/Jerk preferences data 2.csv'
A = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)

# set inference parameters
niter = 1e4
burn = 1000
thin = 4
verbose = 0

# separate into meaningful variable names
x = A[:,1]
yPCGood = A[:,2]
yPCFull = A[:,3]
yPCBad = A[:,4]
yVR = A[:,5]
yTest = A[:,6]

#select just one variable for subsequent analysis:
y = yPCGood

# visualize this variable:
pyplot.close('all')
pyplot.figure()
ax = pyplot.axes()
tsize = 10
markersize = 7
linewidth = 3
ylim = 0.28
ax.plot(x, y, '-', label='Experiment results', color=[0.6,0.6,0.6], linewidth=linewidth)
pyplot.ylim(0,ylim)
xx = np.linspace(0.1, 4, 101)

### MODEL A ###
ax.plot(x, fA(x, 0.27, -0.06), linestyle='-', label='Model A', color='k', linewidth=linewidth)
### MODEL B ###
ax.plot(x, fB(x, 0.23, -0.06, 1), linestyle=':', label='Model B', color='k', linewidth=linewidth)
ax.axvline(x=1, color='k',linestyle='solid',linewidth=0.5,ymax=(0.23 + (-0.06)*1)/ylim )
ax.text(0.4,0.13,'x=b_2',color='black', horizontalalignment='center',rotation=90,size=tsize)
### MODEL C ###
ax.plot(x, fC(x, 0.25, -0.06, 0.5),linestyle='-.', label='Model C', color='k', linewidth=linewidth)
ax.axvline(x=0.5, color='k',linestyle='solid',linewidth=0.5,ymax=(0.25 + (-0.06)*0.5)/ylim )
ax.text(0.9,0.1,'x=c_2',color='black', horizontalalignment='center',rotation=90,size=tsize)

ax.legend()
ax.set_xlabel('Absolute average jerk (deg/s$^3x10^5$)')
ax.set_ylabel('$\mathit{p}$')
pyplot.show()