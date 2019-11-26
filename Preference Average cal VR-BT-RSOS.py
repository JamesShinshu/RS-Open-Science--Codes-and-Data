# -*- coding: utf-8 -*-
import glob, os
import numpy as np
import choix
from matplotlib import pyplot


dirSCRIPT = os.path.dirname(__file__)


filename = os.path.join(dirSCRIPT, 'VR Results')

#filename = '_directory_/VR Results/'
os.chdir(filename)
direc = filename
filename = []
lst_data = []
for file in glob.glob("*.csv"):
    fulldir = '%s/%s' %(direc, file)
    filename.append(fulldir)
    lst_data.append(np.loadtxt(fulldir, delimiter=',',skiprows=1))
    
data = np.asarray(lst_data)
Xdata = ([0.2, 0.5, 1.0, 1.3, 1.9, 2.4, 2.9, 3.4, 3.9])  ##Coesponding AMP jerk values [deg/sec^3] (From 0.0 to 4.0)


nsubjects = data.shape[0]   #number of subjects
ntrials   = data.shape[1]   #number of trials:  72 per subject
### label the AMP conditions:

#################   INPUT VALUES   ######################
amps      = np.linspace(0,4,9)  #all AMP conditions
dataone = 0
datatwo = 1
dAMP = 0.5
#################   INPUT VALUES   ######################

namps     = amps.size       #number of amplitudes

BTdata = []

for i in range( len(data) ):        #Go though each result file
    for ii in range(np.size(data[i],axis=0)):
        if data[i,ii,3] == 1.0:
            BTdata.append( (np.int( (data[i,ii,2])*2 ),np.int( (data[i,ii,1])*2 )) )
        else:
            BTdata.append( (np.int( (data[i,ii,1])*2 ),np.int( (data[i,ii,2])*2 )) )

BT = choix.ilsr_pairwise(namps,BTdata,0.1)
pi = choix.probabilities(list(range(namps)),BT)

pyplot.close('all')
dx = 0.02
textX = 0.6 + dx
textY = 0.13
ax = pyplot.axes()
tsize=12

pyplot.plot(Xdata,pi,'o-',color='black')

pyplot.axvline(0.63,linestyle='--',color='gray')
ax.text(textX,textY,'Human kinematic',color='black',    #### Human kinematics text
        horizontalalignment='center',rotation=90,
        backgroundcolor='w',size=tsize)
        
pyplot.axvline(2.8,linestyle='--',color='gray')
ax.text(2.8,textY+0.03,'Robot kinematic',color='black',  #### Robot kinematics text
        horizontalalignment='center',rotation=90,
        backgroundcolor='w',size=tsize)

pyplot.axvline(Xdata[0],linestyle='-',color='gray')
ax.text(Xdata[0]+dx,textY-0.02,'Minimum jerk\ntrajectory',color='black',  #### Minimum jerk text
        horizontalalignment='center',rotation=90,
        backgroundcolor='w',size=tsize)

pyplot.xlabel('Absolute average jerk (deg/s$^3x10^5$)', size=15)
pyplot.ylabel('$\mathit{p}$', size=15)
pyplot.xlim(0,4.1)
pyplot.tight_layout()
pyplot.show()
