# -*- coding: utf-8 -*-
"""
@author: Frederik Strid
Part of thesis project: ORGANISATION OF THE MOTOR POOL IN BIRDSONG MOTOR CONTROL

This script produces a model of the motor pool based on equations devised 
by Enoka & Fuglevand 2001 (for the motor unit size distribution) and Petersen &
Rostalski 2019 (for the recruitment based on motor unit size). It is intended to
mimic the setup of Adam et al. 2021 with a stimulation paradigm consisting of
pulses linearly increasing in strength over time.

The script is set up in a way where it first generates a motor pool (with all
parameters and added noise) and then runs a loop where samples are taken from the
pool and used to test network analysis as a method for correctly identifying the
distribution of motor units. This is unfortunately unsuccesful :(
    
"""

# %% PACKAGES
import math as mth
import numpy as nmp
from matplotlib import pyplot as plt
import networkx as nwx
from scipy import stats as stt, signal as sig

# %% PARAMETERS OF THE MODEL

#### LOOP PARAMETERS
# how many times the motor pool will be sampled and run through the analysi
# and the size of the samples
iterations = 100
sample_n = 13

## for corr_th
# a sub-loop is run for optimal correlation threshold
iterations_corr = 1000 # number of iterations, determines the decimal precision
step_corr = 10 # step-size for each iteration

## for stimulation
nt = 1000 # length of sTimulation paradigm

#### MU generation
# generating the motor pool. Based on Enoka & Fuglevand 2001 and Adam et al. 2021
n = 30 # number of motor units in the pool
y_1 = 1 # innervation number (IN) of smallest MU. One-to-one innervation here
y_n = 9 # IN of largest MU. Mean max estimate of Adam et al. 2021
R = y_n/y_1 # ratio of largest nad smallest units

y_i = []
for ii in range(n):
    y_i.append(y_1*mth.exp((nmp.log(R)/n)*ii))

# applying random noise to the motor unit sizes for a more natural distribution
rng = nmp.random.default_rng(seed = 2) # normal is 2
noise = rng.standard_normal(n)

# defining motor units from the equation
mu = y_i+noise # adding noise to MU size
mu = [round(i) for i in mu] # converting to integer to avoid decimal fibres

# rounding up motor units of size 0
for ii in range(len(mu)):
    if mu[ii] < 1:
        mu[ii] = 1
        
mu.sort() # re-sorting MUs for later stimulation increase to work

nroi = sum(mu) # a parameter used in the network analysis

## grouping ROIs (MFs) into MUs
# here the individual 'fibres' are generated and categorised by the index of 
# their respective motor unit
roi_mu = []
for iii in range(len(mu)):
    for ii in range(mu[iii]):
        roi_mu.append(iii)

#### distribution plot
# bar plot showing the motor pool size distribution
plt.figure()
plt.plot(y_i, 'C1''--')
plt.bar(range(len(mu)), mu, color ='C0')
plt.xlabel('MU ID')
plt.xlim(-0.75,29.75)
plt.ylabel('MU size (n MF)')

## mu count
# another way of showing the distribution
plt.figure()
plt.hist(mu, bins = range(1,max(mu)+1))

#### STIMULATION
cd_f = 250 # the level of stimulation at which all MUs respond, full recruitment
           # pulling this number out of my arse

# stim that gradually increases but keeps a constant pulse
stim = []
for ii in range(nt):
    stim.append(0)
    if ii % 12 == 0:
        stim.append(ii*((cd_f-15)/(nt)))

stim = stim[:nt] # shortening down stim to length of nt

# noise per MU
roi_mu_noise = []

for iii in range(len(mu)):
    mu_rng = nmp.random.default_rng(seed = iii)
    mu_noise = mu_rng.standard_normal(len(mu))
    for ii in range(mu[iii]):
        roi_mu_noise.append(mu_noise[iii])

# activity signals w activation threshold defined by MU size
# FROM PETERSEN & ROSTALSKI 2019 (eq 1)
# (including cd_f in l59)
a = (nmp.log(100*cd_f))/n
cd_r = [] # recruitment threshold of the MU ii of n

for ii in range(n):
    cd_r.append((mth.exp(a*ii))/100)
    

activity = nmp.zeros((nroi,nt))
for iii in range(nroi):
    for ii in range(nt):
        if stim[ii] > abs(cd_r[roi_mu[iii]]+2*roi_mu_noise[iii]):
            activity[iii,ii] = 1
        else:
            activity[iii,ii] = 0

## noise per MF
# this has quite a big effect on the network analysis' ability to identify
# large motor units correctly. Removing noise gives it very high accuracy
for ii in range(nroi):
    mf_rng = nmp.random.default_rng(seed = ii)
    mf_noise = mf_rng.standard_normal(nt)
    activity[ii,:] = activity[ii,:]+0.125*mf_noise

## normalisation
# so that later plots are easier to read. Can be disabled (#'d) for a more
# accurate representation of the real data
for ii in range(nroi):
    activity[ii,:] = (activity[ii,:]-nmp.min(activity[ii,:]))/ nmp.max(activity[ii,:]-nmp.min(activity[ii,:]))

#### activity signal plot
plt.figure()
plt.imshow(activity)
plt.xlabel('Stimulation time (a.u.)')
plt.ylabel('Fibre ID')

#### colors for plotting activity signals during the loop
# colour-blind friendly colour cycle
cb_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00',
            '#89cff0', '#000000']*n

# %%% visualising stim_th vs stim
stim_th_list = []
for ii in range(nroi):
    stim_th = abs(cd_r[roi_mu[ii]]+2*roi_mu_noise[ii])
    stim_th_list.append(stim_th)

fig, ax1 = plt.subplots()
ax1.plot(stim,'C1')
ax1.set_xlim(-10,1001)
ax1.set_xlabel('stimulation time')
ax1.set_ylabel('activation threshold')

ax2 = ax1.twiny()
ax2.plot(stim_th_list)
ax2.set_xlim(-1.5,106)
ax2.set_xlabel('n of MF')

# %% LOOPING THROUGH GENERATED SAMPLES

analysis = 'ncomm' # for n-cluster optimisation: ncomm
                   # for emd optimisation: emd

opt_mean_corr = []
opt_corr_sd = []
opt_corr_list = []

## for testing increasing sample size, activate this and indent the below
## so that it happens within the larger kk loop
# for kk in range(21):
#     print(' *** Sample iteration: ' + str(kk+1) + ' of ' + str(21))
#     sample_n = 10+kk
#     print('Sample n = ' + str(sample_n))
    
ncomm_to_n = []
ncomm_per_jj = []
optimal_corr_th_per_jj = []

#### loop begins
for jj in range(iterations):
    # picking random sample of MUs
    sample_rng = nmp.random.default_rng(seed = jj)
    sample = sample_rng.choice(range(len(mu)),sample_n, replace = False, shuffle = False)

    # extracting activity of MUs picked by sample
    activity_sample = nmp.zeros_like(activity)
    for ii in range(nroi):
        if roi_mu[ii] in sample:
            activity_sample[ii] = activity[ii]

    # removing empty rows
    activity_sample = activity_sample[~nmp.all(activity_sample == 0, axis=1)]
    
    roi_mu_sample = []
    for ii in roi_mu:
        if ii in sample:
            roi_mu_sample.append(ii)
   
    roi_mu_size = []
    for ii in range(len(roi_mu_sample)):
        roi_mu_size.append(mu[roi_mu_sample[ii]])
    
    #### ANALYSIS I
    # print('Beginning analysis section of iteration ' + str(jj+1) + ' of ' + str(iterations))

    # onset detection
    activity_onset = []
    for ii in activity_sample: 
        peak_x, peak2 = sig.find_peaks(ii, threshold = 0.5)
        activity_onset.append(peak_x[0])

    ncomm_per_ii = []
    optimal_corr_th = []
    corr_th_list = []
    
    if analysis == 'ncomm':
        #### ncomm optimisation
        for ii in range(0,iterations_corr,step_corr):
            corr_mat = nmp.corrcoef(activity_sample)
            nmp.fill_diagonal(corr_mat, 0)
            
            corr_th = ii/iterations_corr
            corr_th_list.append(corr_th)
            
            corr_mat[corr_mat < corr_th] = 0
            g = nwx.from_numpy_array(corr_mat)
            c = list(nwx.community.louvain_communities(g, resolution = 1, seed = 2))
            ncomm = len(c)
            

            # storing ncomm
            ncomm_per_ii.append(ncomm)
        
        # plt.figure()
        # plt.plot(corr_th_list,ncomm_per_ii)
        # plt.xlabel('Correlation threshold')
        # plt.ylabel('n communities')
         
        # finding optimal corr_th (closest to n)
        def closest(lst, K):
                 return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
             
        lst = ncomm_per_ii
        K = sample_n
         
        # saving optimal value for further analysis
        optimal_corr_th = ncomm_per_ii.index(closest(lst, K))/iterations_corr*step_corr
    
    if analysis == 'emd':
        #### emd optimisation
        emd_list = []
        for ii in range(0,iterations_corr,step_corr):
            corr_mat = nmp.corrcoef(activity_sample)
            corr_th = ii/iterations_corr
            # corr_th = 0.49
            corr_mat[corr_mat < corr_th] = 0
            g = nwx.from_numpy_array(corr_mat)
            c = list(nwx.community.louvain_communities(g, resolution = 1, seed = 2))
            ncomm = len(c)
            
            c_mu = [len(i) for i in c]
            roi_c = []
            for iii in range(len(c_mu)):
                for ii in range(c_mu[iii]):
                    roi_c.append(iii)
            
            #  print('Iteration: ' + str(ii+step_corr) + ' of ' + str(iterations_corr) + '. Number of communities: ' + str(ncomm))
            
            # calculate EMD
            emd_list.append(stt.wasserstein_distance(c_mu, mu))
            
         
        # finding optimal corr_th (lowest EMD)
        optimal_corr_th = emd_list.index(min(emd_list))/iterations_corr*step_corr
    
    #### post-optimisation
    print('Optimal correlation threshold is: '+ str(optimal_corr_th))
    optimal_corr_th_per_jj.append(optimal_corr_th)
    
    # and re-doing clustering w optimal correlation
    corr_mat = nmp.corrcoef(activity_sample)
    corr_th = optimal_corr_th
    
    #corr_th = 0.49
    corr_mat[corr_mat < corr_th] = 0
    nmp.fill_diagonal(corr_mat, 0)
    
    g = nwx.from_numpy_array(corr_mat)
    c = list(nwx.community.louvain_communities(g, seed = 2))
    ncomm = len(c)
    
    c_mu = [len(i) for i in c]
    roi_c = []
    for iii in range(len(c_mu)):
        for ii in range(c_mu[iii]):
            roi_c.append(iii)
    
    ncomm_per_jj.append(ncomm)
    ncomm_to_n.append(ncomm/n)
    
    print('Finished iteration ' + str(jj+1) + ' of ' + str(iterations))

    #### comparison plots
    plt.figure(figsize =(40,60))
    plt.subplot(121)
    plt.title('Actual MUs', size = 40)
    for ii in range(len(activity_sample)):
        plt.plot(activity_sample[ii]+ii, color = cb_cycle[roi_mu_sample[ii]])
        plt.text(-20,ii,str(ii))
    plt.subplot(122)
    plt.title('Analysis MUs', size = 40)
    for ii in range(len(activity_sample)):
         plt.plot(activity_sample[ii]+ii, color = cb_cycle[roi_c[ii]])
         plt.text(-20,ii,str(ii))

    

print('Done')

opt_mean_corr.append(nmp.mean(optimal_corr_th_per_jj))
opt_corr_sd.append(nmp.std(optimal_corr_th_per_jj))
opt_corr_list.append(optimal_corr_th_per_jj)

ncomm_to_n.append([i/sample_n for i in ncomm_per_jj])
