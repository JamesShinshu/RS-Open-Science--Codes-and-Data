# RS-Open-Science-Codes-and-Data
Python codes and experimental data used in the paper "Testing for the existence of kinematic uncanny valleys in two-digit grasping"

1. Generating natural gaussian noise to the minimal jerk trajectory using spm1d.
CODE:
  Angle Jerker-RSOS.py
  
2. Measuring the Bradley Terry preference based on the subjects' choices. Use data in PC Results or VR Results. Due to different level of coefficient of consistancy amoung subjects, subject results were organized by Bad, Every, and Good results for the PC Results.
CODES:
  Preference Average cal-BT-RSOS.py (for PC laptop experiment)
  Preference Average cal VR-BT-RSOS.py (for VR experiment)
  
3. Showing an example of Bayesian models for Figure 7.
CODE:
  Bayesian models-RSOS.py
  
4. Calculating the Bayes factor based on the Bradley Terry preference.
CODES:
  Bayesian cal-RSOS.py (for PC laptop experiment)
  Bayesian cal VR-RSOS.py (for VR experiment)
  
Running the codes requires:
* Python 2.7
* NumPy (www.scipy.org) 
* Matplotlib (www.matplotlib.org)
* choix (https://pypi.org/project/choix/)
* pymc (https://pymc-devs.github.io/pymc/)
* spm1d  (www.spm1d.org)
