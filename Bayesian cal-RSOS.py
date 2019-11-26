import numpy as np
from matplotlib import pyplot as plt
import pymc
import os



'''
Define data models
'''
def fA(x, a0, a1):
    return a0 + a1*x

def fB(x, b0, b1, b2):
    f       = b0 + b1*x
    f[x<b2] = b0 + b1*b2
    return f

def fC(x, c0, c1, c2):
    f       = c0 + c1*x
    f[x<c2] = (c0 + 2*c1*c2) -c1*x[x<c2]
    return f


'''
Define observation models and full Bayesian models
'''

def get_modelsA(q0, q1):
    a0   = pymc.Normal("a0", q0, 1e3)
    a1   = pymc.Normal("a1", q1, 1e3)
    err  = pymc.Normal("err", 1e4, 0.1)
    @pymc.deterministic
    def observation(a0=a0, a1=a1):
        return fA(x, a0, a1)
    obsmodel  = pymc.Normal("yA", observation, err, value=y, observed=True)  # observation model
    fullmodel = pymc.Model([observation, a0, a1, obsmodel, err])  # full Bayesian model
    return obsmodel, fullmodel

def get_modelsB(q0, q1):
    b0   = pymc.Normal("b0", q0, 1e3)
    b1   = pymc.Normal("b1", q1, 1e3)
    b2   = pymc.Uniform("b2", x.min(), 1.5 )
    err  = pymc.Normal("err", 1e4, 0.1)
    @pymc.deterministic
    def observation(b0=b0, b1=b1, b2=b2):
        return fB(x, b0, b1, b2)
    obsmodel  = pymc.Normal("yB", observation, err, value=y, observed=True)  # observation model
    fullmodel = pymc.Model([observation, b0, b1, b2, obsmodel, err])  # full Bayesian model
    return obsmodel, fullmodel
    
    
def get_modelsC(q0, q1):
    c0   = pymc.Normal("c0", q0, 1e3)
    c1   = pymc.Normal("c1", q1, 1e3)
    c2   = pymc.Uniform("c2", x.min(), 1.5 )
    err  = pymc.Normal("err", 1e4, 0.1)
    @pymc.deterministic
    def observation(c0=c0, c1=c1, c2=c2):
        return fC(x, c0, c1, c2)
    obsmodel  = pymc.Normal("yC", observation, err, value=y, observed=True)  # observation model
    fullmodel = pymc.Model([observation, c0, c1, c2, obsmodel, err])  # full Bayesian model
    return obsmodel, fullmodel




def get_logp(model, params, posteriors):
    logp       = []
    for q in zip(*posteriors):
        param0 = model.get_node( params[0] )
        param1 = model.get_node( params[1] )
        param0.set_value( q[0] )
        param1.set_value( q[1] )
        if len(params)>2:
            param2 = model.get_node( params[2] )
            param2.set_value( q[2] )
        logp.append( model.logp )
    return np.array(logp)


def set_optim_soln(model, params, posteriors):
    values = posteriors.mean(axis=1)
    param0 = model.get_node( params[0] )
    param1 = model.get_node( params[1] )
    param0.set_value( values[0] )
    param1.set_value( values[1] )
    if len(params)>2:
        param2 = model.get_node( params[2] )
        param2.set_value( values[2] )


def fit_model(model, params):
    mcmc = pymc.MCMC(model)
    mcmc.sample(niter*thin+burn, burn, thin, verbose=verbose, progress_bar=True)
    posteriors = np.array(  [mcmc.trace(s)[:]  for s in params]  )
    logp       = get_logp(model, params, posteriors)
    set_optim_soln(model, params, posteriors)
    return posteriors,logp


def bayes_factor(logp1, logp2):
    # reference: Anand Patil, https://gist.github.com/apatil/179657
    K = np.exp(pymc.flib.logsum(-logp1) - np.log(len(logp1)) - (pymc.flib.logsum(-logp2) - np.log(len(logp2))))
    return K




#(0) Load data:
dirSCRIPT = os.path.dirname(__file__)
fnameCSV  = os.path.join(dirSCRIPT, 'Jerk preferences data.csv')
A         = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)
# separate into meaningful variable names
x         = A[:,1]
yPCGood   = A[:,2]
yPCFull   = A[:,3]
yPCBad    = A[:,4]
yVR       = A[:,5]
yTest     = A[:,6]
# select one for analysis
#y         = yPCGood
y         = yPCFull
#y         = yPCBad
#y         = yVR




#limit to certain x range:
ind = 6
x   = x[:ind]
y   = y[:ind]




#(1) Set inference parameters
niter   = 100000
burn    = 20000
thin    = 1
verbose = 0
# initial parameter values:
q0      = 0.215
q1      = -0.04


#(2) Get observation and full Bayesian models for each data model
obsmodelA,modelA = get_modelsA(q0, q1)
obsmodelB,modelB = get_modelsB(q0, q1)
obsmodelC,modelC = get_modelsC(q0, q1)


#(3) Fit the models using MCMC: (return posterior distributions and logp values)
postsA,logpA     = fit_model(modelA, ['a0', 'a1'])
postsB,logpB     = fit_model(modelB, ['b0', 'b1', 'b2'])
postsC,logpC     = fit_model(modelC, ['c0', 'c1', 'c2'])


#(4) Calculate maximum likelihood values for all Data Model parameters
mlA              = postsA.mean(axis=1)
mlB              = postsB.mean(axis=1)
mlC              = postsC.mean(axis=1)


#(5) Generate lines for fitted models:
xx               = np.linspace(x.min(), x.max(), 101)
predA            = fA(xx, *mlA)
predB            = fB(xx, *mlB)
predC            = fC(xx, *mlC)


#(6) Generate random observations for fitted models:
n     = 20
yhatA = np.array([obsmodelA.rand()  for i in range(n)] )
yhatB = np.array([obsmodelB.rand()  for i in range(n)] )
yhatC = np.array([obsmodelC.rand()  for i in range(n)] )




#(7) Plot fitted models:
plt.close('all')
fig,AX = plt.subplots(1, 3, figsize=(10,3))
ax0,ax1,ax2 = AX.flatten()

for i,(ax,yhat,pred) in enumerate( zip([ax0,ax1,ax2], [yhatA, yhatB, yhatC], [predA,predB,predC]) ):
    h0 = ax.plot(x, y, 'ko-', lw=3, zorder=0)[0]
    h1 = ax.plot(xx, pred, 'r', lw=3, zorder=10)[0]
    h2 = ax.plot(x, yhat.T, '-', color='0.7', lw=0.5, zorder=5)[0]
    
    ax.set_title('Data Model %s' %chr(97+i).upper(), size=16)
    if i==0:
        leg = ax.legend([h0,h1,h2], ['Experimental data', 'Fitted data model', 'Random model realization'])
        plt.setp( leg.get_texts(), size=7 )
plt.show()




#(8) Plot the log p values:  (noticeable separation implies better fits)
# plt.close('all')
plt.figure( figsize=(6,4) )
ax = plt.axes()
ax.plot(logpA)
ax.plot(logpB)
ax.plot(logpC)
ax.legend(['Data Model A', 'Data Model B', 'Data Model C'])
plt.show()



#(9) Compute and report Bayes' factors:
bfAB = bayes_factor(logpA, logpB)
bfBC = bayes_factor(logpB, logpC)
bfAC = bayes_factor(logpA, logpC)
bfBA = bayes_factor(logpB, logpA)
bfCA = bayes_factor(logpC, logpA)
bfCB = bayes_factor(logpC, logpB)
print('')
print('Bayes factor (B:A): %.1f' %bfAB)
print('Bayes factor (C:A): %.1f' %bfAC)
print('Bayes factor (C:B): %.1f' %bfBC)
print('Bayes factor (A:B): %.1f' %bfBA)
print('Bayes factor (A:C): %.1f' %bfCA)
print('Bayes factor (B:C): %.1f' %bfCB)


