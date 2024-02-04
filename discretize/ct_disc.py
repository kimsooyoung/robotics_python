# ref from : https://aleksandarhaber.com/python-control-systems-library-tutorial-2-define-state-space-models-generate-step-response-discretize-and-perform-basics-operations/

import matplotlib.pyplot as plt
import control as ct
import numpy as np
 
# this function is used for plotting of responses 
# it generates a plot and saves the plot in a file 
 
# xAxisVector - time vector
# yAxisVector - response vector
# titleString - title of the plot
# stringXaxis - x axis label 
# stringYaxis - y axis label
# stringFileName - file name for saving the plot, usually png or pdf files
 
def plottingFunction(xAxisVector,yAxisVector,titleString,stringXaxis,stringYaxis,stringFileName):
    plt.figure(figsize=(8,6))
    plt.plot(xAxisVector,yAxisVector, color='blue',linewidth=4)
    plt.title(titleString, fontsize=14)
    plt.xlabel(stringXaxis, fontsize=14)
    plt.ylabel(stringYaxis,fontsize=14)
    plt.tick_params(axis='both',which='major',labelsize=14)
    plt.grid(visible=True)
    # plt.savefig(stringFileName,dpi=600)
    plt.show()

A=np.array([[0, 1],[-4, -2]])
B=np.array([[0],[1]])
C=np.array([[1,0]])
D=np.array([[0]])
sysStateSpace=ct.ss(A,B,C,D)
 
print(sysStateSpace)

# # Response and Basic Computations
# timeVector=np.linspace(0, 5, 100)
# timeReturned, systemOutput = ct.step_response(sysStateSpace,timeVector)

# # plot the step response
# plottingFunction(timeReturned,systemOutput,titleString='Step Response',stringXaxis='time [s]' , stringYaxis='Output',stringFileName='stepResponse.png')


# discretization time
sampleTime=0.1
# discretize the system dynamics
# method='zoh' - zero order hold discretization
# method='bilinear' - bilinear discretization
sysStateSpaceDiscrete=ct.sample_system(sysStateSpace, sampleTime, method='zoh') 

# check if the system is in discrete-time
sysStateSpaceDiscrete.isdtime()
print(sysStateSpaceDiscrete)

# compute the step response
timeVector2=np.linspace(0,5,np.int32(np.floor(5/sampleTime)+1))
timeReturned2, systemOutput2 = ct.step_response(sysStateSpaceDiscrete, timeVector2)

# plot the step response
plottingFunction(timeReturned2,systemOutput2,titleString='Step Response',stringXaxis='time [s]' , stringYaxis='Output',stringFileName='stepResponse.png')
