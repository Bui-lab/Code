#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""
from collections import Counter

import Const
import json
import matplotlib.image as mpimg
import numpy as np
# Import pandas for data saving
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams, animation
from numpy import  zeros

from Analysis_tools import angles_

def saveToJSON(filename, content):
    jscontent = json.dumps(content)
    f = open(filename,"w")
    f.write(jscontent)
    f.close()

def readFromJSON(filename):
    f = open(filename,"r")
    jscontent = f.read()    
    content = json.loads(jscontent)
    f.close()
    return content
    

# The keys of the LeftValues and RightValues dictionaries are the cell names, like "IC", "MN" etc
# The column names in the csv file start with Left_IC, Right_MN, etc and end with cell number
# There are no gaps between the columns of the same neuron type and side.
# For each cell type, first left cells, than right cells are saved
def saveToCSV(filename, Time, LeftValues, RightValues):

    #check for input accuracy - a file name and Time array need to be provided
    if filename is None or Time is None or\
        dict(LeftValues).keys() != dict(RightValues).keys():
        return

    #Sim_data is the pandas dataframe that will be used to save into a .csv
    Sim_data = pd.DataFrame(index=Time)
    for cellname in dict(LeftValues).keys():
        groupValues = LeftValues[cellname]
        numcells = len(groupValues)
        for j in range(0, numcells):
            header_name = 'Left_' + cellname + str(j)
            col_df = pd.DataFrame(index = Time, data = groupValues[j], columns = [header_name])
            Sim_data = pd.concat([Sim_data, col_df], axis= 1)
        groupValues = RightValues[cellname]
        for k in range(0, numcells):
            header_name = 'Right_' + cellname + str(k)
            col_df = pd.DataFrame(index = Time, data = groupValues[k], columns = [header_name])
            Sim_data = pd.concat([Sim_data, col_df], axis= 1)

    Sim_data.to_csv(filename, index_label ='Time')

# cell_names is the list of neurons, like "IC", "MN" etc
# Assumption 1: The column names in the csv file start with Left_IC, Right_MN, etc and end with cell number
# Assumption 2: There are no gaps between the columns of the same neuron type and side
def readFromCSV(filename, cell_names):
    if filename is None:
        return
    read_data = pd.read_csv(filename)

    data_top = list(read_data.columns.values.tolist())

    read_sim = np.ascontiguousarray(read_data)
    read_sim = np.transpose(read_sim)
    Time = read_sim[0]

    LeftValues = dict()
    RightValues = dict()
    for nt in cell_names:
        #find the columns that start with Left_[neuron type name]
        #enumerate adds the indices to data_top: x[0] refers to the index and x[1] refers to the values 
        #matchingcols is a list of index and value tuple, which can be reached by [,0] and [,1] respectively
        matchingcols = list(filter(lambda x: x[1].startswith("Left_" + nt), enumerate(data_top)))
        if len(matchingcols) == 0:
            continue
        next_start = matchingcols[0][0]
        next_end = matchingcols[-1][0]
        LeftValues[nt] = read_sim[next_start:next_end+1]

        matchingcols = list(filter(lambda x: x[1].startswith("Right_" + nt), enumerate(data_top)))
        if len(matchingcols) == 0:
            continue
        next_start = matchingcols[0][0]
        next_end = matchingcols[-1][0]
        RightValues[nt] = read_sim[next_start:next_end+1]

    return Time, LeftValues, RightValues


# nMuscle: the number of somites
# dt: the discretization time
def saveAnimation(filename, nMuscle, VLMuscle, VRMuscle, Time, dt):

    if filename is None or nMuscle is None or \
        VLMuscle is None or VRMuscle is None or \
        Time is None or dt is None:
        return

    # Uncomment the line below if ffmpeg.exe is not already in your system environment variable PATH
    # plt.rcParams['animation.ffmpeg_path'] = "c:/Program Files/ffmpeg/ffmpeg.exe" #Change if ffmpeg.exe is in another location
    
    # Calculate the number of time points
    nmax = len(Time)
    ani = angles_(Time, nMuscle, nmax, VRMuscle, VLMuscle, dt)
    ani.save(filename)  #, fps=30)#, extra_args=['-vcodec', 'libx264'])


#This function creates the multipanel animation combining musculoskeletal model with cell firing
#Assumption: leftValues and rightValues are dictionaries holding the membrane potential values of neurons and muscle cells
#The key of the dictionary is the type of neuron ("IC", "MN", etc) or "Muscle"
#colors is the dictionary holding the color
def multipanel_anim(Time, nmax, leftValues, rightValues, leftColors, rightColors, dt, imgfile, title):
    
    plt.rc('lines', linewidth=Const.MULTIPANEL_LINEWIDTH) 
    # Change default font to Arial
    rcParams['font.sans-serif'] = "Arial"
    # Then, "ALWAYS use sans-serif fonts"
    rcParams['font.family'] = "sans-serif"

    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.bf'] =  'Arial:italic:bold'

    figheight = 9
    figwidth = 15
    plotindex_angles = 133 # On a 1x3 image, 3rd position
    plotindex_diagram = 131 # On a 1x3 image, 1st position
    numofcols = 3
    firingplotinc = 2

    if (imgfile is None):
        figwidth = 10
        plotindex_angles = 122 # On a 1x2 image, 2nd position
        numofcols = 2
        firingplotinc = 1

    # Declare figure and subplot
    fig = plt.figure(figsize=(figwidth, figheight))
    
    fig_angles = fig.add_subplot(plotindex_angles)    # musculoskeletal model
    fig_sublist = dict()
    left_firing = dict()
    right_firing = dict()

    nMuscle = len(leftValues['Muscle'][:, 0])
    VLMuscle = leftValues['Muscle']
    VRMuscle = rightValues['Muscle']
    numoffiring = len(list(filter(lambda x: x != 'Muscle', dict(leftValues).keys())))

    for k in dict(leftValues).keys():
        if k != "Muscle":
            figsub = fig.add_subplot(numoffiring, numofcols, firingplotinc)
            firingplotinc += numofcols
            fig_sublist[k] = figsub
        
            #Declare the various left and right traces to be plotted
            firing, = figsub.plot([], [], lw=1, color = leftColors[k])
            left_firing[k] = firing
            firing, = figsub.plot([], [], lw=1, color = rightColors[k])
            right_firing[k] = firing
            

    fig_angles.set_title(title)

    if imgfile is not None:
        # insert double coiling diagram
        fig_diagram = fig.add_subplot(plotindex_diagram)
        img = mpimg.imread(imgfile)
        fig_diagram.imshow(img)
        fig_diagram.axis('off')

    Muscle_angles, = fig_angles.plot([], [], 'o-', lw=3, color = 'Black')
    Muscle_angles_highlight, = fig_angles.plot([], [], 'o-', lw=3, color = 'Red')

    # Allocating arrays for velocity and position
    vel = np.zeros((nMuscle, nmax))
    pos = np.zeros((nMuscle, nmax))
    
    # Setting constants and initial values for vel. and pos.
    khi = 3.0  #damping cste , high khi =0.5/ low = 0.1
    w0 = 2.5 #2.5  #20Hz = 125.6
    vel0 = 0.0
    pos0 = 0.0
    #Wd = w0
    
    for k in range (0,nMuscle):
        vel[k,0] = vel0    #Sets the initial velocity
        pos[k,0] = pos0    #Sets the initial position
        pos[nMuscle-1,0] = 0.0
        for i in range(1,nmax):
        
            vel[k,i] = -(w0**2)*pos[k,i-1]*dt + vel[k,i-1]*(1-(2*dt*khi*w0)) + 0.1*VRMuscle[k,i-1]*dt - 0.1*VLMuscle[k,i-1]*dt
            pos[k,i] = dt*vel[k,i-1] + pos[k,i-1]
    
    ### DYNAMIC PLOTING
    
    x = np.zeros((nMuscle,nmax))
    y = np.zeros((nMuscle,nmax))
    
    for i in range (0,nmax):
        x[0,i] = 0
        y[0,i] = 0
        pos[0,i] = 0
        for k in range (1,nMuscle):
            pos[k,i] = pos[k-1,i] + pos[k,i]
            
            x[k,i] = x[k-1,i] + np.sin(pos[k,i])
            y[k,i] = y[k-1,i] - np.cos(pos[k,i])
    
    #Declare x and y-axis limits for the various figures
    fig_angles.grid()
    fig_angles.set_ylim(-15, 5)
    fig_angles.set_xlim(-10, 10)
    for k in fig_sublist.keys():
        figsub = fig_sublist[k]
        figsub.set_ylim(-80, 20)
        figsub.set_xlim(0, nmax*dt)

    # declare time text
    time_template = 'time = %.1fms'
    time_text = fig_angles.text(0.05, 0.1, '', transform=fig_angles.transAxes)
    fig_angles.legend()
    fig_angles.set_xticks([])
    
    for k in fig_sublist.keys():
        figsub = fig_sublist[k]
        #Set up legend
        leg=figsub.legend(handles=[left_firing[k], right_firing[k]], labels=['L '+ k,'R '+ k], loc='upper right', 
            handlelength=Const.MULTIPANEL_LINELENGTH, fontsize=Const.MULTIPANEL_SMALLER_SIZE)
        leg.legendHandles[0].set_color(leftColors[k])
        leg.legendHandles[1].set_color(rightColors[k])
        for line in leg.get_lines():
            line.set_linewidth(Const.MULTIPANEL_LINEWIDTH)

        figsub.set_ylabel(r"$\mathbf{Vm}$" + " (mV)", fontsize= Const.MULTIPANEL_SMALL_SIZE, fontweight=Const.MULTIPANEL_FONT_STYLE) #y-axis title
        figsub.set_ylim([Const.MULTIPANEL_LOWER_Y, Const.MULTIPANEL_UPPER_Y]) #y-axis limits
        # Remove borders
        figsub.spines['top'].set_visible(False)
        figsub.spines['right'].set_visible(False)
        figsub.spines['bottom'].set_visible(False)
        figsub.spines['left'].set_visible(False)
        #Set up ticks
        figsub.tick_params(axis='both', which='both', length=0) 
        for item in ([figsub.title, figsub.xaxis.label, figsub.yaxis.label] +
                figsub.get_xticklabels() + figsub.get_yticklabels()):
            item.set_fontsize(Const.MULTIPANEL_SMALL_SIZE)
        figsub.set_yticks([i*50 + -50 for i in range(0,2)])
        figsub.set_xticks([i*5000 for i in range(0,5)])
        figsub.set_xlabel('Time (ms)', fontsize= Const.MULTIPANEL_SMALL_SIZE, fontweight='bold') #x-axis title
        figsub.set_xlim([Time[0], Time[-1]]) #x-axis limits
    
    #This function initializes the animation
    def init():
        Muscle_angles.set_data([], [])
        for k in left_firing.keys():
            left_firing[k].set_data([], [])
            right_firing[k].set_data([], [])
        time_text.set_text('')
     
    #This function drives the animation by updating every time point
    def animate(i):
        
        thisx = [x[k,i] for k in range(nMuscle)]
        thisy = [y[k,i] for k in range(nMuscle)]
        
        Muscle_angles.set_data(thisx, thisy)
        Muscle_angles_highlight.set_data(x[3,i], y[3,i])
        time_text.set_text(time_template % (Time[i]))

        for k in left_firing.keys():
            left_firing[k].set_data(Time[0:i], leftValues[k][3, 0:i])
            right_firing[k].set_data(Time[0:i], rightValues[k][3, 0:i])

        return Muscle_angles, left_firing, right_firing, time_text
     
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(Time), 10), 
                                  interval=10, blit=False, init_func=init)
        
    plt.show()
    return ani

#leftValues and rightValues are dictionaries holding membrane potentials of different cell types, multiple cells
#if onSamePlot = 1: left and right values displayed on the same plot
#if there is only one neuron type to plot, left will be plotted on the top, right will be plotted on the bottom
#if there are multiple neuron types to plot, the height is the height for each row
#colorMode: 0 -> Left and Right colors will be used
#colorMode: 1 -> Neuron type based colors will be used
def plotProgress(tstart, tend, timeArray, leftValues, rightValues, onSamePlot = False, width = 15, height = 5, colorMode = 0):

    if dict(leftValues).keys() != dict(rightValues).keys():
        return 0
    
    numofplots = len(dict(leftValues).keys())
    x_axis = timeArray[tstart: tend]

    numofrows = numofplots
    numofcols = 1 if onSamePlot else 2
    if numofplots == 1 and not onSamePlot:
        numofrows = 2
        numofcols = 1
    fig, ax = plt.subplots(numofrows, numofcols, sharex=True, figsize=(width, height * numofrows))

    rowind = 0
    for k in dict(leftValues).keys():

        nCells = len(leftValues[k][:, 0])
        listLeft = leftValues[k]
        listRight = rightValues[k]
        colorLeft = Const.IPSI_COLOR_MAPS[k] if colorMode == 1 else Const.IPSI_COLOR_MAPS['Left']
        colorRight = Const.CONTRA_COLOR_MAPS[k] if colorMode == 1 else Const.CONTRA_COLOR_MAPS['Right']

        if numofplots == 1 and onSamePlot:
            ax.plot([0], [0], c=colorLeft(0.5))
            ax.set_title(k)
            for k in range (0, nCells):
                ax.plot(x_axis, listLeft[k,tstart: tend], c=colorLeft((k+1)/nCells)) # adding a color gradiant, darker color -> rostrally located
                ax.plot(x_axis, listRight[k,tstart: tend], c=colorRight((k+1)/nCells))
        elif numofplots == 1:
            ax[0].plot([0], [0], c=colorLeft(0.5))
            ax[0].set_title(k)
            for k in range (0, nCells):
                ax[0].plot(x_axis, listLeft[k,tstart: tend], c=colorLeft((k+1)/nCells)) # adding a color gradiant, darker color -> rostrally located
                ax[1].plot(x_axis, listRight[k,tstart: tend], c=colorRight((k+1)/nCells))
        elif numofcols == 1:
            ax[0].plot([0], [0], c=colorLeft(0.5))
            ax[rowind].set_title(k)
            for k in range (0, nCells):
                ax[rowind].plot(x_axis, listLeft[k,tstart: tend], c=colorLeft((k+1)/nCells)) # adding a color gradiant, darker color -> rostrally located
                ax[rowind].plot(x_axis, listRight[k,tstart: tend], c=colorRight((k+1)/nCells))
            rowind += 1
        else:
            colRight = 0 if onSamePlot else 1
            ax[0, 0].plot([0], [0], c=colorLeft(0.5))
            ax[rowind, 0].set_title(k)
            for k in range (0, nCells):
                ax[rowind, 0].plot(x_axis, listLeft[k,tstart: tend], c=colorLeft((k+1)/nCells)) # adding a color gradiant, darker color -> rostrally located
                ax[rowind, colRight].plot(x_axis, listRight[k,tstart: tend], c=colorRight((k+1)/nCells))
            rowind += 1
        
    plt.xlabel('Time (ms)')
    plt.xlim([timeArray[tstart], timeArray[tend] + 1])
    plt.show()
    return fig, ax

#import subprocess #Required to play sound in Mac - uncomment appropriate lines in PlaySound() function
import winsound

def PlaySound():
    #settings for Windows
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    #subprocess.call(['afplay', 'Sound.wav']) #Put a wave file of your choice for the sound
