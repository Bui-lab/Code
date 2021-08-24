#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 24 2020

@authors: Yann Roussel and Tuan Bui
"""

from Izhikevich_class import * # Where definition of single cell models are found based on Izhikevich models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
from random import *
from statistics import mode
from scipy.signal import find_peaks
import math

#Function to calculate auto-correlation of x
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result[int(result.size/2):]
    return result/np.max(result)

#Function to calculate cross-correlation of x and y
def Xcorr(x, y):
    result = np.correlate(x, y, mode='full')
    result = result[int(result.size/2):]
    return result/np.max(result)

#Function to calculate cross-correlation of x and y. The difference between Xcorr and Xcorr_LR is that the former returns half of the 
# correlation result (only the positive time delay values)
def Xcorr_LR(x, y):
    result = np.correlate(x, y, mode='full')
    return result/np.max(result)

#Function that returns the normalized cross-correlation
def norm_Xcorr(x, y):
    # Normalised_CrossCorr = 1/N * sum{ [x(n) - mean(x)]* [y(n) - mean(y)] }/ (sqrt(var(x)*var(y))
    x_adjusted = x - np.mean(x)
    y_adjusted = y - np.mean(y)
    result = np.correlate(x_adjusted, y_adjusted, mode='full')/(len(x)*np.sqrt(np.var(x)*np.var(y)))
    return result

#This function calculates the angle of the musculoskeletal model based on the VRMuscle and VLMuscle data
# It also produces an animated plot of the musculoskeletal model
def angles_(Time, nMuscle, nmax, VRMuscle, VLMuscle, dt, title = ''):
    # Allocating arrays for velocity and position
    vel = np.zeros((nMuscle, nmax))
    pos = np.zeros((nMuscle, nmax))
    
    # Setting constants and initial values for vel. and pos.
    khi = 3.0  #damping cste , high khi =0.5/ low = 0.1
    w0 = 2.5 #2.5  #20Hz = 125.6
    vel0 = 0.0
    pos0 = 0.0
    Wd = w0
    
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-nMuscle - 5, 5))
    ax.grid()
    ax.set_title(title)
    
    line, = ax.plot([], [], 'o-', lw=5, label="Muscle")
    time_template = 'time = %.1fms'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.legend()
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    #This function will animate the musculoskeletal model based on updates at every time point i
    def animate(i):
        
        thisx = [ x[k,i] for k in range(nMuscle)]
        thisy = [  y[k,i] for k in range(nMuscle)]
        
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (Time[i]))
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(Time), 100), #animation.FuncAnimation(fig, animate, np.arange(1, len(Time), 10), for faster animation
                                  interval=100, blit=False, init_func=init) # interval = 10
    
    #ani.save('CPG_ani_BnG.mpeg', writer="ffmpeg")
    #ani.save("./results_test/" + name + ".mp4")#, fps=30)#, extra_args=['-vcodec', 'libx264'])
    
    #from matplotlib.animation import FFMpegWriter. The two lines below may be useful if you are having trouble with the animation
    #writer = FFMpegWriter(fps=1000, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save("Single coiling.mp4", writer=writer)
    
    plt.show()
    return ani

#This function calculates the heatmap of the body angles as calculated by VLMuscle and VRMuscle
def Heatmap(VLMuscle, VRMuscle, dt=0.1, vmin_=-0.5, vmax_=0.5, ymin=0, ymax=1000):

    nmax = len(VLMuscle[0,:])
    nMuscle = len(VLMuscle[:,0])
    
    # Allocating arrays for velocity and position
    vel = np.zeros((nMuscle, nmax))
    pos = np.zeros((nMuscle, nmax))
    
    # Setting constants and initial values for vel. and pos.
    khi = 3.0  #damping cste , high khi =0.5/ low = 0.1
    w0 = 2.5 #2.5  #20Hz = 125.6
    vel0 = 0.0
    pos0 = 0.0
    Wd = w0
    
    for k in range (0,nMuscle):
        vel[k,0] = vel0    #Sets the initial velocity
        pos[k,0] = pos0    #Sets the initial position
        pos[nMuscle-1,0] = 0.0
        for i in range(1,nmax):
        
            vel[k,i] = -(w0**2)*pos[k,i-1]*dt + vel[k,i-1]*(1-(2*dt*khi*w0)) + 0.1*VRMuscle[k,i-1]*dt - 0.1*VLMuscle[k,i-1]*dt
            pos[k,i] = dt*vel[k,i-1] + pos[k,i-1]
    
    pos2 = pos.transpose()
    FFT = zeros((nmax,nMuscle-1))
    FFT2 = zeros((int(nmax/2),nMuscle-1))
    
    figure()
    plt.pcolormesh(np.arange(nMuscle)/nMuscle, np.arange(nmax)*dt, pos2, cmap=plt.cm.bwr, vmin=vmin_, vmax=vmax_)
    plt.title('Local Body Angle', fontsize = 14)
    plt.ylabel('Time (ms)', fontsize = 14)
    #plt.ylim(0,1000)
    plt.ylim(ymin, ymax)
    plt.xlabel('Body position', fontsize = 14)
    plt.colorbar()
    plt.show()
    
    ##Fourier
    
    Fs = 10000.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,1,Ts) # time vector
    
    n = nmax # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    
    freqs = np.fft.fftfreq(int(nMuscle/2), Ts)
    idx = np.argsort(freqs)
    
    for k in range(0, nMuscle-1):
        
        FFT[:,k] = np.fft.fft(pos2[:,k])/n
        FFT2[:,k] = FFT[range(int(n/2)),k]
        FFT2[:,k] = sqrt(FFT2[:,k]*(FFT2[:,k].conjugate()))
        
    figure()
    plt.pcolormesh(np.arange(nMuscle)/nMuscle, frq, FFT2, cmap=plt.cm.bwr)
    plt.title('FFT Magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0,100)
    plt.xlabel('Body position')
    plt.colorbar()
    plt.show()

    return(pos2, FFT2)

#This function smoothes y by convolving using a box
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#This function calculates the start, end and duration of swimming episode, as defined by a threshold
def detect_event(VLMuscle, VRMuscle, threshold):
    
    X = np.sum(VLMuscle, axis=0) + np.sum(VRMuscle, axis=0)
    X = smooth(X, 500) #convolve with a step 50 ms wide
    Xt = Time[np.where(X > threshold)]
    
    plt.plot(Time, X)
    plt.axhline(y=threshold, ls='--', c='r')
    plt.xlabel('Time (ms)')
    plt.ylabel('Integrated motor output (arbitrary units)')
    
    plt.rcParams.update({'font.size': 22})
    
    if not any(Xt):
        end =[]
        start=[]
        duration = []
    else:
        end = Xt[[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]+[True]]
        start = Xt[[True]+[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]]
        duration = end - start
        
    return start, end, duration

#This function calculates the start, end and duration of swimming episode, as defined by a threshold
def detect_event(VLMuscle, VRMuscle, Time, Threshold):
    
    X = np.sum(VLMuscle, axis=0) + np.sum(VRMuscle, axis=0)
    X = smooth(X, 500) #convolve with a step 50 ms wide
    Xt = Time[np.where(X > Threshold)]
    
    plt.plot(Time, X)
    plt.axhline(y=Threshold, ls='--', c='r')

    if not any(Xt):
        end =[]
        start=[]
        duration = []
    else:
        end = Xt[[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]+[True]]
        start = Xt[[True]+[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]]
        duration = end - start
        
    
    return start, end, duration

#This function calculates the start, end and duration of swimming episode, as defined by a threshold. Does not plot the result
def detect_event_no_plot(VLMuscle, VRMuscle, Time, Threshold):
    
    X = np.sum(VLMuscle, axis=0) + np.sum(VRMuscle, axis=0)
    X = smooth(X, 500) #convolve with a step 50 ms wide
    Xt = Time[np.where(X > Threshold)]
    
    if not any(Xt):
        end =[]
        start=[]
        duration = []
    else:
        end = Xt[[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]+[True]]
        start = Xt[[True]+[Xt[i+1] - Xt[i] > 0.2 for i in range(len(Xt)-1)]]
        duration = end - start
        
    
    return start, end, duration

#This plot takes the VLMN and VRMN output and calculates the muscle output based upon R, C and weight of MN to Muscle connection 
# that are arguments of the function. Returns VLMuscle and VRMuscle
def recalc_muscle_ouptut(VLMN, VRMN, Time, dt, nMN, nMuscle, R, C, weight_MN_Muscle):

    nmax = len(Time)
    
    L_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt=dt, x=5.0+1.6*i,y=-1) for i in range(nMN)]
    R_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt=dt, x=5.0+1.6*i,y=1) for i in range(nMN)]
    
    L_Muscle = [ Leaky_Integrator(R, C, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    R_Muscle = [ Leaky_Integrator(R, C, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    
    ## Declare Synapses
       
    L_glusyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)] 
    R_glusyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)]

    VLMuscle = zeros((nMuscle, nmax))
    VRMuscle = zeros((nMuscle, nmax))
    
    #Ach
    LSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    RSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    
    #Ach
    LW_MN_Muscle = zeros((nMN,nMuscle))
    RW_MN_Muscle = zeros((nMN,nMuscle))
    
    # residuals
    resLMuscle = zeros((nMuscle,2))
    resRMuscle = zeros((nMuscle,2))
    
    # Calculating MN_muscle weights
    MN_Muscle_syn_weight = weight_MN_Muscle 
    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (L_Muscle[l].x-1<L_MN[k].x< L_Muscle[l].x+1):         #this connection is segmental
                LW_MN_Muscle[k,l] = MN_Muscle_syn_weight
            else:
                LW_MN_Muscle[k,l] = 0.0
    
    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (R_Muscle[l].x-1<R_MN[k].x< R_Muscle[l].x+1):         #it is segmental
                RW_MN_Muscle[k,l] = MN_Muscle_syn_weight  
            else:
                RW_MN_Muscle[k,l] = 0.0
    
    for k in range (0, nMuscle):
        resLMuscle[k,:] = L_Muscle[k].getNextVal(0,0)
        VLMuscle[k,0] = resLMuscle[k,0]
        
        resRMuscle[k,:] = R_Muscle[k].getNextVal(0,0)
        VRMuscle[k,0] = resRMuscle[k,0]
    
    for t in range (0, nmax):
        Time[t]=dt*t
                
        for k in range (0, nMN):
            for l in range (0, nMuscle):
                LSyn_MN_Muscle[nMuscle*k+l,:] = L_glusyn_MN_Muscle[nMuscle*k+l].getNextVal(VLMN[k,t-10], VLMuscle[l,t-1], LSyn_MN_Muscle[nMuscle*k+l,1], LSyn_MN_Muscle[nMuscle*k+l,2])
                RSyn_MN_Muscle[nMuscle*k+l,:] = R_glusyn_MN_Muscle[nMuscle*k+l].getNextVal(VRMN[k,t-10], VRMuscle[l,t-1], RSyn_MN_Muscle[nMuscle*k+l,1], RSyn_MN_Muscle[nMuscle*k+l,2])
                
        ## Calculate membrane potentials
            
        for k in range (0, nMuscle):
            IsynL = sum(LSyn_MN_Muscle[nMuscle*l+k,0]*LW_MN_Muscle[l,k] for l in range (0, nMN))
            IsynR = sum(RSyn_MN_Muscle[nMuscle*l+k,0]*RW_MN_Muscle[l,k] for l in range (0, nMN))
                
            resLMuscle[k,:] = L_Muscle[k].getNextVal(resLMuscle[k,0], IsynL) #the last term is to add variability but equals 0 if sigma = 0
            VLMuscle[k,t] = resLMuscle[k,0]
            
            resRMuscle[k,:] = R_Muscle[k].getNextVal(resRMuscle[k,0], IsynR) #the last term is to add variability but equals 0 if sigma = 0
            VRMuscle[k,t] = resRMuscle[k,0]
            
    return VLMuscle, VRMuscle


def phase_relationship(X, Y, Time):
    X_peak = find_peaks(X, height=[-20,], threshold=None, distance=None, prominence=1, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    Y_peak = find_peaks(Y, height=[-20,], threshold=None, distance=None, prominence=1, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    
    return phase_rel

#This function calculates tail beat frequency based upon crossings of y = 0 as calculated from the body angles calculated
# by VRMuscle and VLMuscle
def calc_tail_beat_freq(VRMuscle, VLMuscle, nmax, dt, lower_bound, upper_bound, delay):
    
    ### Calculate angles

    vmin_=-0.1 
    vmax_=0.1

    ymin = 0
    ymax = 5000

    nMuscle = len(VRMuscle)
    
    # Allocating arrays for velocity and position
    vel = np.zeros((nMuscle, nmax))
    pos = np.zeros((nMuscle, nmax))

    # Setting constants and initial values for vel. and pos.
    khi = 3.0  #damping cste , high khi =0.5/ low = 0.1
    w0 = 2.5 #2.5  #20Hz = 125.6
    vel0 = 0.0
    pos0 = 0.0
    Wd = w0

    for k in range (0,nMuscle):
        vel[k,0] = vel0    #Sets the initial velocity
        pos[k,0] = pos0    #Sets the initial position
        pos[nMuscle-1,0] = 0.0
        for i in range(1,nmax):

            vel[k,i] = -(w0**2)*pos[k,i-1]*dt + vel[k,i-1]*(1-(2*dt*khi*w0)) + 0.1*VRMuscle[k,i-1]*dt - 0.1*VLMuscle[k,i-1]*dt
            pos[k,i] = dt*vel[k,i-1] + pos[k,i-1]

    pos2 = pos.transpose()
    angle = pos2

    ### Measure x and y coordinates based on angles at each segment

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

    # We will only use the tip of the tail to determine tail beats (if the x coordinate of the tip is smaller (or more negative)
    # than the lower bound or if the x coordinate of the tip is greater than the upper bound, then detect as a tail beat
    tail_tip_x = x[nMuscle-1, :]

    def check_all_in(the_lower_bound, the_upper_bound, alist):
        in_list = True
        for i in range(0, len(alist)): 
            if alist[i] < the_lower_bound or alist[i] > the_upper_bound:
                in_list = False
        return in_list

    Between_episodes = 1
    num_tail_beats=[]
    interbeat_interval=[]
    start=[]
    beat_times=[]
    side = 0
    LEFT = -1
    RIGHT = 1
    cross = 0
    for i in range(0, len(tail_tip_x)-10):

        if check_all_in(lower_bound, upper_bound, tail_tip_x[i:i+delay]):
            Between_episodes = 1
            if side == LEFT or side == RIGHT: # if we are coming out of an episode
                num_tail_beats.append(cross)
            side = 0
            cross = 0

        if Between_episodes == 1 and tail_tip_x[i] < lower_bound:  # beginning an episode on the left
            cross += 1
            side = LEFT
            Between_episodes = 0
            start.append(i*0.1)
            beat_times.append(i*0.1)
            interbeat_interval.append(math.nan)
            First = True
        elif Between_episodes == 1 and tail_tip_x[i] > upper_bound: # beginning an episode on the right
            cross += 1
            side = RIGHT
            Between_episodes = 0
            start.append(i*0.1)
            beat_times.append(i*0.1)
            interbeat_interval.append(math.nan)
            First = True
        
        # During an episode
        if tail_tip_x[i] < lower_bound and side == RIGHT:
            cross += 1
            side = LEFT
            interbeat_interval.append(i*0.1 - beat_times[-1])
            beat_times.append(i*0.1)
        elif tail_tip_x[i] > upper_bound and side == LEFT:
            cross += 1
            side = RIGHT
            interbeat_interval.append(i*0.1 - beat_times[-1])
            beat_times.append(i*0.1)
        
    return num_tail_beats, interbeat_interval, start, beat_times