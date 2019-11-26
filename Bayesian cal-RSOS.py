import numpy as np
from matplotlib import pyplot
import pymc
import os

dirSCRIPT = os.path.dirname(__file__)

fnameCSV = os.path.join(dirSCRIPT, 'Jerk preferences data.csv')
A = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)

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
    f[x<c2] = (c0 + 2*c1*c2) -c1*x[x<c2]
    return f

# set inference parameters
niter = 1e6
burn = 100000
thin = 6
verbose = 0

# separate into meaningful variable names
x = A[:,1]
yPCGood = A[:,2]
yPCFull = A[:,3]
yPCBad = A[:,4]
yVR = A[:,5]
yTest = A[:,6]

#select just one variable for subsequent analysis:
y = yVR


q0 = 0.215
q1 = -0.04

#Model A
a0 = pymc.Normal("a0", q0, 1e3)
a1 = pymc.Normal("a1", q1, 1e3)
errA = pymc.Normal("errA", 1e4, 0.1)

@pymc.deterministic
def predictionA(a0=a0, a1=a1):
    return fA(x,a0,a1)

ymodelA = pymc.Normal("yA", predictionA, errA, value=y, observed=True)

#Model B
b0 = pymc.Normal("b0", q0, 1e3)
b1 = pymc.Normal("b1", q1, 1e3)
b2 = pymc.Uniform("b2", x.min(), x.max() )
errB = pymc.Normal("errB", 1e3, 0.1)

@pymc.deterministic
def predictionB(b0=b0, b1=b1, b2=b2):
    return fB(x, b0, b1, b2)
ymodelB = pymc.Normal("yB", predictionB, errB, value=y, observed=True)

#Model C
c0 = pymc.Normal("c0", q0, 1e3)
c1 = pymc.Normal("c1", q1, 1e3)
c2 = pymc.Uniform("c2", x.min(), x.max() )
errC = pymc.Normal("errC", 1e3, 0.1)


@pymc.deterministic
def predictionC(c0=c0, c1=c1, c2=c2):
    return fC(x, c0, c1, c2)
ymodelC = pymc.Normal("yC", predictionC, errC, value=y, observed=True)

# construct PyMC models:
modelA = pymc.Model([predictionA, a0, a1, ymodelA, errA])
modelB = pymc.Model([predictionB, b0, b1, b2, ymodelB, errB])
modelC = pymc.Model([predictionC, c0, c1, c2, ymodelC, errC])

# conduct inference separately for each model:
mcmcA = pymc.MCMC(modelA)
mcmcA.sample(niter*thin+burn, burn, thin, verbose=verbose, progress_bar=True)

mcmcB = pymc.MCMC(modelB)
mcmcB.sample(niter*thin+burn, burn, thin, verbose=verbose, progress_bar=True)

mcmcC = pymc.MCMC(modelC)
mcmcC.sample(niter*thin+burn, burn, thin, verbose=verbose, progress_bar=True)

aa0 = mcmcA.trace('a0')[ : ]
aa1 = mcmcA.trace('a1')[ : ] 

bb0 = mcmcB.trace('b0')[ : ]
bb1 = mcmcB.trace('b1')[ : ] 
bb2 = mcmcB.trace('b2')[ : ]

cc0 = mcmcC.trace('c0')[ : ]
cc1 = mcmcC.trace('c1')[ : ] 
cc2 = mcmcC.trace('c2')[ : ]


def get_logpA():
    traces = [mcmcA.trace(s) for s in ['a0', 'a1']]
    logp = []
    for aa0, aa1 in zip(*traces):
        a0.set_value(aa0)
        a1.set_value(aa1)
        logp.append(modelA.logp)
    return np.array(logp)

def get_logpB():
    traces = [mcmcB.trace(s) for s in ['b0', 'b1', 'b2']]
    logp = []
    for bb0, bb1, bb2 in zip(*traces):
        b0.set_value(bb0)
        b1.set_value(bb1)
        b2.set_value(bb2)
        logp.append(modelB.logp)
    return np.array(logp)

def get_logpC():
    traces = [mcmcC.trace(s) for s in ['c0', 'c1', 'c2']]
    logp = []
    for cc0, cc1, cc2 in zip(*traces):
        c0.set_value(cc0)
        c1.set_value(cc1)
        c2.set_value(cc2)
        logp.append(modelC.logp)
    return np.array(logp)

logpA = get_logpA()
logpB = get_logpB()
logpC = get_logpC()


def bayes_factor(logp1, logp2):
    # reference: Anand Patil, http://gist.githum.com/179657
    K = np.exp(pymc.flib.logsum(-logp1) - np.log(len(logp1)) - (pymc.flib.logsum(-logp2) - np.log(len(logp2))))
    return K
    

bfAB = bayes_factor(logpA, logpB)
bfBC = bayes_factor(logpB, logpC)
bfAC = bayes_factor(logpA, logpC)
bfBA = bayes_factor(logpB, logpA)
bfCA = bayes_factor(logpC, logpA)
bfCB = bayes_factor(logpC, logpB)

print('')
print('Bayes factor (A vs. B): %.1f' %bfAB)
print('Bayes factor (A vs. C): %.1f' %bfAC)
print('Bayes factor (B vs. A): %.1f' %bfBA)
print('Bayes factor (B vs. C): %.1f' %bfBC)
print('Bayes factor (C vs. A): %.1f' %bfCA)
print('Bayes factor (C vs. B): %.1f' %bfCB)


# visualize this variable:
pyplot.close('all')
pyplot.figure()
ax = pyplot.axes()
ax = pyplot.subplot(141)
ax.hist(logpA, label='Model A')
ax.hist(logpB, label='Model B')
ax.hist(logpC, label='Model C')
ax.legend()
ax.set_xlabel('Log probability')
ax.set_ylabel('Frequency')
ax = pyplot.subplot(142)
ax.hist(aa0, label='a0')
ax.hist(aa1, label='a1')
ax = pyplot.subplot(143)
ax.hist(bb0, label='b0')
ax.hist(bb1, label='b1')
ax.hist(bb2, label='b2')
ax = pyplot.subplot(144)
ax.hist(cc0, label='c0')
ax.hist(cc1, label='c1')
ax.hist(cc2, label='c2')

pyplot.show()