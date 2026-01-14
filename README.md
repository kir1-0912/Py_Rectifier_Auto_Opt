# Py_Rectifier_Auto_Opt
Automated Optimization Framework for RF Circuit Front-End Rectifier Circuits

Allocate_EMX.py: Invoke the command line via instructions to call the Spectre simulation netlist .scs file to record data and return target values. The simulation values are read from the .raw file generated after the simulation to calculate the target values. In the process, use the bisection method to find the matching resistor that achieves the highest efficiency.Change the optimization target by adjusting the contents of the fre & pin list.

Includes common optimization algorithms such as MORBO, NSGA, RVEA, and MOPSO, and the MORBO folder also contains STuRBO.
