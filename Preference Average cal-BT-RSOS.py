# -*- coding: utf-8 -*-
import glob, os
import numpy as np
import choix
from matplotlib import pyplot


dirSCRIPT = os.path.dirname(__file__)


PCresults = os.path.join(dirSCRIPT, 'PC Results')

nameoffile = os.path.join(PCresults, 'Good results')                #HACKberry hand test (good data only)
nameoffile = os.path.join(PCresults, 'Every results')               #HACKberry hand test (Full data)
nameoffile = os.path.join(PCresults, 'Bad results')                #HACKberry hand test (bad data only)

os.chdir(nameoffile)
direc = nameoffile
os.chdir(nameoffile)
direc = nameoffile
filename = []
lst_data = []

for file in glob.glob("*.npy"):
    fulldir = '%s/%s' %(direc, file)
    filename.append(fulldir)
    lst_data.append(np.load(fulldir))
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


#(1) Stack the subject arrays (and label variable columns):
### stacking data vertically like this makes it much easier to algorithmically process lots of data
A         = np.vstack(data)   #648 x 4 array
SUBJ      = np.array([ntrials*[i]   for i in range(nsubjects)]).flatten()   #subject labels
LEFT      = A[:,0]  #left animation jerk amplitude
RIGHT     = A[:,1]  #right animation jerk amplitude
CHECK     = A[:,3]  #selected video (0 = left, 1 = right)

crt = np.zeros((namps,namps))
BTdata = []

for i in range( len(data) ):        #Go though each result file
    for ii in range(np.size(data[i],axis=0)):
        if data[i,ii,3] == 1.0:
            BTdata.append( (np.int( (data[i,ii,datatwo])*2 ),np.int( (data[i,ii,dataone])*2 )) )
        else:
            BTdata.append( (np.int( (data[i,ii,dataone])*2 ),np.int( (data[i,ii,datatwo])*2 )) )

###---Calculate Statistical values---###

### Measuring the general agreement between every subjects ###

BT = choix.ilsr_pairwise(namps,BTdata,0.1)
pi = choix.probabilities(list(range(namps)),BT)

pyplot.close('all')
dx = 0.02
textX = 0.6 + dx
textY = 0.15
ax = pyplot.axes()
tsize=12

pyplot.plot(Xdata,pi,'o-',color='k')

#pyplot.axvline(0.63,linestyle='--',color='gray')
#ax.text(textX,textY,'Human kinematic',color='black',    #### Human kinematics text
#        horizontalalignment='center',rotation=90,
#        backgroundcolor='w',size=tsize)
#        
#pyplot.axvline(2.8,linestyle='--',color='gray')
#ax.text(2.8,textY+0.03,'Robot kinematic',color='black',  #### Robot kinematics text
#        horizontalalignment='center',rotation=90,
#        backgroundcolor='w',size=tsize)
#
#pyplot.axvline(Xdata[0],linestyle='-',color='gray')
#ax.text(Xdata[0]+dx,textY-0.02,'Minimum jerk\ntrajectory',color='black',  #### Minimum jerk text
#        horizontalalignment='center',rotation=90,
#        backgroundcolor='w',size=tsize)

pyplot.xlabel('Absolute average jerk (deg/s$^3x10^5$)', size=15)
pyplot.ylabel('$\mathit{p}$', size=15)
pyplot.xlim(0,4.1)
pyplot.ylim(0,0.28)
pyplot.tight_layout()
pyplot.show()