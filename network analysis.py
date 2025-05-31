# -*- coding: utf-8 -*-
"""
@author: Frederik Strid
Part of thesis project: ORGANISATION OF THE MOTOR POOL IN BIRDSONG MOTOR CONTROL

This script runs the network analysis described in the thesis on the actual data instead of the model.
It can run any specimen and electrode but only one combination at a time. It is set up for both n-cluster
and EMD optimisaion of the correlation threshold. It is broadly similar to the overview of all electrodes
and model scripts but combines the network analysis of the model with the actual data of the overview.

Parts of the script (mainly the calculation of DFF per ROI and some of the 
network analysis) is based on a tutorial from CalTech (Andreev 2023),
which can be found here:
https://focalplane.biologists.com/2023/10/27/analyzing-calcium-imaging-data-using-python/

This script is set up to run data that is not publicly available and
is for viewing purposes only
"""

# %%% packages
import tifffile
import numpy as nmp
from matplotlib import pyplot as plt
from scipy import signal as sig, stats as stt
import networkx as nwx
import pandas as pd

# %% SPECIMEN SELECTION
# select the specimen and electrode to run
# note that not all specimens and electrode combinations are available
# GW65: purple, orange, grey
# GW64: grey
# GW55: purple, orange, green
# 7391: purple, grey, green

## stitching
specimen = '7391' # choose between GW65, Gw64, GW55, or 7391
electrode = 'purple'

# %%% SETTING UP THE DATA
# This section loads in the selected specimen + electrode, stitches it together, loads the
# fibre coordinates and the reference motor units

# specimen GW65 is annotated
# this process is very similar to the overview script except it takes 
# one electrode at a time
if specimen == 'gw65':
    #### gw65
    x1 = 195 # x coordinate for edge of overlap for 65r2
    x2 = 22 # x coordinate for edge of overlap for 65r1

    ## moco series
    print('Loading ' + specimen + '_' + electrode + ' MOCO data') 
    # this script runs only one electrode at a time
    if electrode == 'purple':
        trig1 = 94 # the time point where the stimulation paradigm begins
        trig2 = 35
              
        # motion corrected data files
        moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series024_ch01_R2_purple_el_moco.tif'
        moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series008_ch01_R1_purple_el_moco.tif'
    
        # convert from .tiff to array
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        # define dimensions (used for stitching)
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
        
        # pre-initialising the array for the stitched specimen
        # DFF = change in fluorescence (delta_F/F)
        data_dff = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
        
        # loading in the two halves in their respective parts of the array
        # in a way so that they match up spatially and temporally
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,:,x2:]
    
    # next electrode, follows the same procedure
    elif electrode == 'orange':
        trig1 = 79
        trig2 = 83
        
        moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series029_ch01_R2_orange_el_moco.tif'
        moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series011_ch01_R1_orange_el_moco.tif'
    
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
        
        data_dff = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:,:,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:,x2:]
        
    elif electrode == 'grey':
        trig1 = 87
        trig2 = 79
        
        moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series030_ch01_R2_grey_el_moco.tif'
        moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series013_ch01_R1_grey_el_moco.tif'
    
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
        
        data_dff = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,:,x2:]
        
    else:
        print('Invalid electrode identifier for ' + str(specimen))


    ## field-stimulation
    trig1 = 36-10 # the time point where the stimulation paradigm begins
    trig2 = 57-10 
    
    # motion-corrected field-stimulation data
    muscle_moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series028_ch01_R2_muscle_moco.tif'
    muscle_moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series015_ch01_R1_muscle_moco.tif'
    
    # again, convert .tiff to array
    muscle_dat1 = tifffile.imread(muscle_moco1)
    muscle_dat2 = tifffile.imread(muscle_moco2)
    
    # same procedure as for the electrode dff data
    nt1, ny1, nx1 = nmp.shape(muscle_dat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(muscle_dat2[:,:,:])
    
    # aligning
    muscle_moco = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
    muscle_moco[:,:,:x1] = muscle_dat1[trig1:,:,:x1]
    muscle_moco[:,:,x1:] = muscle_dat2[trig2:nt2-17,:,x2:]
    
    ## averaged dff data, used as a background in many plots
    print('Loading ' + specimen + '_' + electrode + ' average image data')
    avg1 = 'Data/210303_f_gw65/AVG_210303_f_gw65_MSD_R_Series032_ch01_R2 for_moco.tif'
    avg2 = 'Data/210303_f_gw65/AVG_210303_f_gw65_MSD_R_Series017_ch01_R1_for moco.tif'
    
    avgdat1 = tifffile.imread(avg1)
    avgdat2 = tifffile.imread(avg2)
    
    ny1, nx1 = nmp.shape(avgdat1[:,:]) 
    ny2, nx2 = nmp.shape(avgdat2[:,:]) 
    
    avg_moco = nmp.zeros((ny1, (nx1+nx2)-(x2+(200-x1))))
    avg_moco[:,:x1] = avgdat1[:,:x1]
    avg_moco[:,x1:] = avgdat2[:,x2:]
    
        
    ## brightness for the background
    brightness = 500


    ## MF coordinates
    # loaded in for each half and will be assembled later
    print('Loading ' + specimen + '_' + electrode + ' mask data') 
    mask_r1 = pd.read_csv('Data/210303_f_gw65/210303_f_gw65_MSD_R_Series017_RAW_ch01_mask_for_python.csv')
    mask_r2 = pd.read_csv('Data/210303_f_gw65/210303_f_gw65_MSD_R_Series032_RAW_ch01_mask_for_python.csv')

    ## known MUs
    # the indexes for the coordinates of the identified reference MUs
    mu_1 = [158]
    mu_2 = [156,157,19,69]
    mu_3 = [33]
    mu_4 = [60]
    mu_5 = [68,75]
    mu_6 = [39,63,28]
    mu_7 = [146]
    mu_8 = [147]
    mu_9 = [15,74,93,94,99,120,121,127,129]
    mu_10 = [40]
    mu_11 = [64]
    mu_12 = [8,11,84,101,113,81]
    mu_13 = [91,92]
    
    mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6,mu_7,mu_8,mu_9,mu_10,mu_11,mu_12,mu_13] # list of lists
    mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7+mu_8+mu_9+mu_10+mu_11+mu_12+mu_13
    
    # used when doing signal-to-noise ratio based filtering later
    sd_mult = 5.5

#
elif specimen == 'gw64': 
    #### gw64
    # for more detailed annotation see specimen GW65
    x1 = 198
    x2 = 40 
       
    print('Loading ' + specimen + '_' + electrode + ' MOCO data')
    if electrode == 'grey':
        trig1 = 57
        trig2 = 21
        
        moco1 = 'Data/210305_f_gw64/210305_f_gw64_Series028_ch01_grey_el_moco_fixed.tif'
        moco2 = 'Data/210305_f_gw64/210305_f_gw64_Series034_ch01_R2_grey_el_moco.tif'
        
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
        
        data_dff = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,:,x2:]
          
    else:
        print('Invalid electrode identifier for ' + str(specimen))                           

    
    ## muscle stim
    trig1 = 47-10
    trig2 = 48-10
    
    muscle_moco1 = 'Data/210305_f_gw64/210305_f_gw64_Series029_ch01_muscle_moco.tif'
    muscle_moco2 = 'Data/210305_f_gw64/210305_f_gw64_Series039_ch01_R2_muscle_moco.tif'
    
    muscle_dat1 = tifffile.imread(muscle_moco1)
    muscle_dat2 = tifffile.imread(muscle_moco2)
    
    nt1, ny1, nx1 = nmp.shape(muscle_dat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(muscle_dat2[:,:,:])
    
    # aligning
    muscle_moco = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
    muscle_moco[:,:,:x1] = muscle_dat1[trig1:,:,:x1]
    muscle_moco[:,:,x1:] = muscle_dat2[trig2:nt2-11,:,x2:]
    
    
    ## avg image
    print('Loading ' + specimen + '_' + electrode + ' average image data')
    avg1 = 'Data/210305_f_gw64/AVG_210305_f_gw64_MDS_L_Series030_ch01_R1_ for_moco.tif'
    avg2 = 'Data/210305_f_gw64/AVG_210305_f_gw64_MDS_L_Series037_ch01_R2_for_moco.tif'
        
    avgdat1 = tifffile.imread(avg1)
    avgdat2 = tifffile.imread(avg2)
    
    ny1, nx1 = nmp.shape(avgdat1[:,:]) 
    ny2, nx2 = nmp.shape(avgdat2[:,:]) 
    
    # aligning
    avg_moco = nmp.zeros((ny1, (nx1+nx2)-(x2+(200-x1))))
    avg_moco[:,:x1] = avgdat1[:,:x1]
    avg_moco[:,x1:] = avgdat2[:,x2:]
    
    
    ## brightness for bg
    brightness = 130    
    
    
    ## mask
    print('Loading ' + specimen + '_' + electrode + ' mask data')
    mask_r1 = pd.read_csv('Data/210305_f_gw64/210305_f_gw64_MDS_L_Series037_RAW_ch01_mask_for_python.csv')
    mask_r2 = pd.read_csv('Data/210305_f_gw64/210305_f_gw64_MDS_L_Series030_RAW_ch01_mask_for_python.csv')
    
    
    ## known MUs
    mu_1 = [128,159,116,81,100,102]
    mu_2 = [101]
    mu_3 = [95]
    mu_4 = [68,73,74,75,76]
    
    mu = [mu_1,mu_2,mu_3,mu_4]
    mu_list = mu_1+mu_2+mu_3+mu_4
    sd_mult = 5

elif specimen == 'gw55': 
    #### gw55
    # this specimen is weird because there is a major issue with the spatial
    # alignment of the two halfs of the data set
    # because all of the previously (by Adam et al) identified MUs are only on
    # one half, this is the one that is used for analysis
    # the possibility also exists for using both halves but it is only
    # for the visual impression
    x1 = 123
    x2 = 0
    
    ## moco series
    print('Loading ' + specimen + '_' + electrode + ' MOCO data')
    if electrode == 'purple':
        trig1 = 46
        trig2 = 46
        
        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series006_ch01_purple_el_moco.tif'
        moco2 = 'Data/210304_f_gw55/210304_f_gw55_Series030_ch01_R2_purple_el_moco.tif'
    
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
        data_dff = nmp.zeros((nt2-trig2, ny1-16, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:ny1-16,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,16:,x2:]
        
    elif electrode == 'purple_r1':

        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series006_ch01_purple_el_moco.tif'
        mocodat1 = tifffile.imread(moco1)
        
        data_dff = mocodat1
    
    elif electrode == 'green':
        trig1 = 66
        trig2 = 33
        
        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series009_ch01_green_el_moco.tif'
        moco2 = 'Data/210304_f_gw55/210304_f_gw55_Series032_ch01_R2_green_el_moco.tif'
    
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
        data_dff = nmp.zeros((nt2-trig2, ny1-16, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:ny1-16,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,16:,x2:]
        
    elif electrode == 'green_r1':

        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series009_ch01_green_el_moco.tif'
        mocodat1 = tifffile.imread(moco1)
        
        data_dff = mocodat1
    
    elif electrode == 'orange':
        trig1 = 59
        trig2 = 34
        
        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series014_ch01_orange_el_moco.tif'
        moco2 = 'Data/210304_f_gw55/210304_f_gw55_Series033_ch01_R2_orange_el_moco.tif'
    
        mocodat1 = tifffile.imread(moco1)
        mocodat2 = tifffile.imread(moco2)
        
        nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
        nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
        data_dff = nmp.zeros((nt2-trig2, ny1-16, (nx1+nx2)-(x2+(200-x1))))
        
        data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:ny1-16,:x1]
        data_dff[:,:,x1:] = mocodat2[trig2:,16:,x2:]
        
    elif electrode == 'orange_r1':
        
        moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series014_ch01_orange_el_moco.tif'
        mocodat1 = tifffile.imread(moco1)
        
        data_dff = mocodat1
        
    else:
        print('**Invalid electrode identifier for ' + str(specimen) + '**') 

    
    ## muscle stim
    ## only R1 works
    
    muscle_moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series019_ch01_muscle_moco.tif'
    muscle_dat1 = tifffile.imread(muscle_moco1)
    
    muscle_moco = muscle_dat1
    

    ## avg image
    print('Loading ' + specimen + '_' + electrode + ' average image data')
    avg1 = 'Data/210304_f_gw55/AVG_210304_f_gw55_MDS_R_Series021_ch01_for_moco.tif'
    # avg2 = 'Data/210304_f_gw55/AVG_210304_f_gw55_MDS_R_Series029_ch01_R2_for_moco.tif'
    
    avgdat1 = tifffile.imread(avg1)
    # avgdat2 = tifffile.imread(avg2)
    
    # ny1, nx1 = nmp.shape(avgdat1[:,:]) 
    # ny2, nx2 = nmp.shape(avgdat2[:,:])
    
    # avg_moco = nmp.zeros((ny1-16, (nx1+nx2)-(x2+(200-x1))))
    # avg_moco[:,:x1] = avgdat1[:ny1-16,:x1]
    # avg_moco[:,x1:] = avgdat2[16:,x2:]
    avg_moco = avgdat1 # only for running gw55 half
        
    ## brightness for bg
    brightness = 1000

    
    ## mask
    print('Loading ' + specimen + '_' + electrode + ' mask data')
    mask_r1 = pd.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series029_RAW_ch01_mask_for_python.csv')
    mask_r2 = pd.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series021_RAW_ch01_mask_for_python.csv')
    
    
    ## known MUs
    mu_1 = [31]
    mu_2 = [66]
    mu_3 = [24,25,72,14]
    mu_4 = [3,53,62]
    mu_5 = [33,52,41]
    mu_6 = [58]
    mu_7 = [51,71]
    
    mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6, mu_7]
    mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7
    
    sd_mult = 4.5
    
elif specimen == '7391':
   #### 7391
   x1 = 200 
   x2 = 45
    
   ## moco series    
   print('Loading ' + specimen + '_' + electrode + ' MOCO data')
   if electrode == 'purple':
       trig1 = 57
       trig2 = 88

       moco1 = 'Data/210303_f_7391/210303_f_7391_Series042_ch01_R2_purple_el_moco.tif'        
       moco2 = 'Data/210303_f_7391/210303_f_7391_Series021_ch01_R1_purple_el_moco.tif'
        
       mocodat1 = tifffile.imread(moco1)
       mocodat2 = tifffile.imread(moco2)
       
       nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       # these scans are not properly aligned on the y axis so this is adjusted
       # during the stitching by adding empty rows to mocodat2 for later removal 
      
       mocodat2 = nmp.append(nmp.zeros((nt2,32,nx2)),mocodat2,axis=1)
       # redefining the shape parameters to include this
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       data_dff = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
       
       data_dff[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
       data_dff[:,:,x1:] = mocodat2[trig2:,:ny2-32,x2:]
       
   elif electrode == 'grey':
       trig1 = 53
       trig2 = 117
       
       moco1 = 'Data/210303_f_7391/210303_f_7391_Series045_ch01_R2_grey_el_moco.tif'   
       moco2 = 'Data/210303_f_7391/210303_f_7391_Series030_ch01_R1_grey_el_moco.tif'   
       
       mocodat1 = tifffile.imread(moco1)
       mocodat2 = tifffile.imread(moco2)
       
       nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       # these scans are not properly aligned on the y axis so this is adjusted
       # during the stitching by adding empty rows to mocodat2 for later removal
       
       mocodat2 = nmp.append(nmp.zeros((nt2,32,nx2)),mocodat2,axis=1)
       # redefining the shape parameters to include this
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       data_dff = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
       
       data_dff[:,:,:x1] = mocodat1[trig1:,:,:x1]
       data_dff[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:ny2-32,x2:]
       
   elif electrode == 'green':
       trig1 = 66
       trig2 = 118
       
       moco1 = 'Data/210303_f_7391/210303_f_7391_Series047_ch01_R2_green_el_moco.tif'   
       moco2 = 'Data/210303_f_7391/210303_f_7391_Series025_ch01_R1_green_el_moco.tif'   
       
       mocodat1 = tifffile.imread(moco1)
       mocodat2 = tifffile.imread(moco2)
       
       nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       # these scans are not properly aligned on the y axis so this is adjusted
       # during the stitching by adding empty rows to mocodat2 for later removal
       
       mocodat2 = nmp.append(nmp.zeros((nt2,32,nx2)),mocodat2,axis=1)
       # redefining the shape parameters to include this
       nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
       
       data_dff = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
       
       data_dff[:,:,:x1] = mocodat1[trig1:,:,:x1]
       data_dff[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:ny2-32,x2:]       

   else:
       print('Invalid electrode identifier for ' + str(specimen))     
   

   ## muscle stim
   trig1 = 153-10 
   trig2 = 29-10 
   
   muscle_moco1 = 'Data/210303_f_7391/210303_f_7391_Series067_ch01_R2_muscle.tif'
   muscle_moco2 = 'Data/210303_f_7391/210303_f_7391_Series032_ch01_R1_muscle_rigidbody.tif'
   
   muscle_dat1 = tifffile.imread(muscle_moco1)
   muscle_dat2 = tifffile.imread(muscle_moco2)
   
   nt1, ny1, nx1 = nmp.shape(muscle_dat1[:,:,:]) 
   nt2, ny2, nx2 = nmp.shape(muscle_dat2[:,:,:])
   
   # aligning
   muscle_moco = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
   muscle_moco[:,:,:x1] = muscle_dat1[trig1:nt1-37,:,:x1]
   muscle_moco[:,:,x1:] = muscle_dat2[trig2:,:,x2:]
   
   ## avg image
   print('Creating ' + specimen + '_' + electrode + ' average image data')
   avg_moco = nmp.mean(data_dff, axis = 0)
   
   
   ## brightness for bg
   brightness = 600
   

   ## mask
   print('Loading ' + specimen + '_' + electrode + ' mask data')
   mask_r1 = pd.read_csv('Data/210303_f_7391/R1_210303_f_7391_MSD_R_Series0XX_RAW_ch01_mask_for_python.csv')
   mask_r2 = pd.read_csv('Data/210303_f_7391/R2_210303_f_7391_MSD_R_Series0XX_RAW_ch01_mask_for_python.csv')
   
   ## known MUs
   mu_1 = [14,117]
   mu_2 = [19,79,109]
   mu_3 = [64,92,97,98,106,110,167,171]
   mu_4 = [13,113,115,124,173]
   mu_5 = [17,16]
   mu_6 = [147,103,94]
   mu_7 = [184]
   mu_8 = [126]
   mu_9 = [132] 
   mu_10 = [148]
   mu_11 = [129,149]
   mu_12 = [128]
   mu_13 = [35]
   mu_14 = [47]
   mu_15 = [158,159,176]
   mu_16 = [73,74]
   mu_17 = [53,77,63]
   mu_18 = [11,172]

   mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6,mu_7,mu_8,mu_9,mu_10,mu_11,mu_12,mu_13,mu_14,mu_15,mu_16,mu_17,mu_18]
   mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7+mu_8+mu_9+mu_10+mu_11+mu_12+mu_13+mu_14+mu_15+mu_16+mu_17+mu_18
   
   sd_mult = 8
    
else:
    print('Invalid specimen identifier')

print('Generating parameters from ' + specimen + '_' + electrode + ' data') 

# setting some parameters based the specimen
bg = avg_moco
nt, ny, nx = nmp.shape(data_dff[:,:,:]) 
nt_m, ny_m, nx_m = nmp.shape(muscle_moco[:,:,:]) 

#### plotting mask

# extracting individual XY coords from the mask
x_r1,y_r1 = mask_r1.get('x'), mask_r1.get('y') # for merged
x_r2,y_r2 = mask_r2.get('x'), mask_r2.get('y')

# removing overlapping XY coords
if specimen == 'gw55' or specimen == '7391': 
    # because gw55 is stitched right on left not left on right
    for ii in range(0, len(x_r1)): 
        if x_r1[ii] < nx1-x1:
            x_r1[ii] = x_r1[ii]+nx
else:
    for ii in range(0, len(x_r1)):
        if x_r1[ii] < x2:
            x_r1[ii] = x_r1[ii]+nx 
            # moved out of frame and removed when removing 
            # close to edge
 
# shifting right to align with r1
x_r1 = x_r1+x1-x2

# shifting the vertical alignment for gw55 and 7391
# because for both specimen the original scans do not align in the Y-plane
if specimen == 'gw55':
    y_r1 = y_r1-16
if specimen == '7391':
    y_r1 = y_r1+32

# making a combined list of coordinates now that both sides are aligned
# and overlap has been removed
x = pd.concat([x_r2, x_r1], ignore_index = True)
y = pd.concat([y_r2, y_r1], ignore_index = True)

n_mf = len(x) # total number of MFs, including dead and inactive

w = 2 # the 'radius' (square area) around the coords defining the ROIs

# removing POIs too close to edge to generate ROIs within boundaries
for ii in range(0, len(x)):
    if x[ii]+w > nx:
        x = x.drop(ii)
        y = y.drop(ii)
    elif x[ii]-w < 0:
        x = x.drop(ii)
        y = y.drop(ii)

# removing holes in the index as caused by .drop()
x = x.set_axis(range(0,len(x)))
y = y.set_axis(range(0,len(y)))

nroi = len(x)

# plotting
# plt.figure(figsize=(12,6))
# plt.text(10,10, 'specimen = ' + specimen, color = 'w')
# plt.text(10,15, 'n = ' + str(nroi), color = 'w')
# plt.text(10,20,'w = ' + str(w), color = 'w')
# plt.xticks(ticks = ())
# plt.yticks(ticks = ())

# bg = avg_moco
# plt.imshow(bg[:,:], vmax = brightness) # avg moco bg

# plotting mask ROIs with numbering

# for i in range(0,len(x)):
#     plt.scatter(x[i],y[i], marker="$"+str(i)+"$", color = 'w', linewidths = 0.5)

#### detecting live fibres
print('Detecting live fibres')

if specimen != '7391':
    # again, specimen 7391 lacks the field-stim MOCO for detecting alive fibres
    
    # pre-initialising arrays
    activity_muscle = nmp.zeros((nroi, nt_m))
    
    # calculating DFF for field-stimulation data
    for ii in range(nroi):
        # dF/F for each pixel in the w*w area of the ROI
        xy_roi_dff = muscle_moco[:,y[ii] + nmp.arange(-w,w),:][:,:,x[ii] + nmp.arange(-w,w)]
        # averaging spatially for the ROI
        xy_roi_dff_mean = nmp.mean(xy_roi_dff,(1,2))
        # storing the mean DFF of each roi
        activity_muscle[ii,:] = xy_roi_dff_mean
    
    # removing NaN values (fibres too close to edge to have fit in the w*w area)
    # should be superfluous as this was done when making the mask
    activity_muscle = activity_muscle[~nmp.isnan(activity_muscle).any(axis = 1)]     
    
    
    ## removing the dead fibres based on correlaiton values
    corr_mat = nmp.corrcoef(activity_muscle)
    
    # plotting raw correlation matrix
    # plt.figure(figsize = (15,10))
    # plt.subplot(121)
    # plt.imshow(corr_mat)
    # plt.title('Raw correlation matrix')
    
    # removing negative correlations (dead fibres)
    for ii in range(0, len(corr_mat)):
        if nmp.mean(corr_mat[ii,:]) < 0: # in one dimension
                corr_mat[ii,:] = 0     
            
        if nmp.mean(corr_mat[:,ii]) < 0: # and the other
                corr_mat[:,ii] = 0
    
    # saving which are dead and alive
    dead = nmp.where(nmp.all(corr_mat == 0, axis = 1))[0]
    alive = [i for i in list(range(nroi)) if i not in dead]
    
    # coords for alive/dead fibres
    x_alive = x[alive].to_list()
    x_dead = x[dead].to_list()
    y_alive = y[alive].to_list()
    y_dead = y[dead].to_list()
    
# cannot be done for specimen 7391 but it still needs a list of coordinates
# for the next part of the analysis
elif specimen == '7391':
    alive = list(range(nroi))
    dead = []
    x_alive = x.copy()
    y_alive = y.copy()

plt.figure(figsize=(12,6))
plt.text(10,10, 'specimen = ' + specimen, color = 'w')
plt.text(10,15, 'n alive = ' + str(len(alive)), color = 'w')
plt.text(10,20, 'n dead = ' + str(len(dead)), color = 'w')
plt.xticks(ticks = ())
plt.yticks(ticks = ())
plt.imshow(bg[:,:], vmax = brightness) # avg moco bg

# plotting new ROIs
for i in range(len(alive)):
    plt.scatter(x_alive[i],y_alive[i], marker="$"+str(i)+"$", color = 'w', linewidths = 0.5)
for i in range(len(dead)):
    plt.scatter(x_dead[i],y_dead[i], marker="$"+str(i)+"$", color = 'r', linewidths = 0.5)

#### calculating dff
# this is the dF/F calculation for the electrode data, which will later be used
# to stimulated fibres
# it follows the same basic procedure as the DFF for the field-stimulation except
# now it's done for all of the electrodes
# (this is usually the most taxing part of this script)

# pre-initialising array
dff = nmp.zeros_like(data_dff) 

# defining the baseline fluorescence
F0 = nmp.mean(data_dff[:,:,:], axis = (0))

# dF/F = (F_t-F0)/F0
for ii in range(0, nt):
    dff[ii,:,:] = nmp.array((data_dff[ii,:,:]-F0)/F0)


## saving the DFF per ROI (alive)

# pre-initialising array
activity = nmp.zeros((len(alive), nt))

for ii in range(len(alive)):
    xy_roi_dff = dff[:,y_alive[ii] + nmp.arange(-w,w),:][:,:,x_alive[ii] + nmp.arange(-w,w)]
    xy_roi_dff_mean = nmp.mean(xy_roi_dff,(1,2))
    activity[ii,:] = xy_roi_dff_mean

# removing NaN values (fibres too close to edge to have fit in the w*w area)
activity = activity[~nmp.isnan(activity).any(axis = 1)]    

#### SNR
print('Detecting active signals')

# pre-inisatialising array and list for coordinates
activity_filtered = nmp.zeros_like(activity)
x_filtered = []
y_filtered = []
filtered_index = []

# using signal-to-noise ratio to identify response to stimulation
for ii in range(len(activity)):
    # calculating SNR parameters
    noise_mean = nmp.mean(activity[ii,:])
    noise_std = nmp.std(activity[ii,:])
    # finding signals with peaks that stand out of the noise
    peak_x, peak2 = sig.find_peaks(activity[ii], threshold = noise_mean+sd_mult*noise_std)
    if not any(peak_x) == False:
    # saving the signals with peaks in them
        activity_filtered[ii] = activity[ii]
        x_filtered.append(x_alive[ii])
        y_filtered.append(y_alive[ii])
        filtered_index.append(ii)

x_filtered = pd.Series(x_filtered)
y_filtered = pd.Series(y_filtered)

# removing empty rows
activity_filtered = activity_filtered[~nmp.all(activity_filtered == 0, axis=1)]


# normalising for readability ONLY USE FOR PLOTTING NOT ANALYSIS
activity_filtered_norm = nmp.zeros_like(activity_filtered)
for ii in range(len(activity_filtered)):
    activity_filtered_norm[ii] = (activity_filtered[ii] - nmp.min(activity_filtered[ii])) / \
        nmp.max(activity_filtered[ii] - nmp.min(activity_filtered[ii]))

## plotting
range_filtered = range(len(activity_filtered))
plt.figure(figsize=(20, 40))
plt.title(str(specimen) + '_' + str(electrode) + ' using SNR-filter', size = 40)
for ii in range(len(activity_filtered)):
    plt.plot(activity_filtered_norm[ii]*10+16*range_filtered[ii], 'C0')
    # plt.text(-250, range_filtered[ii]*16, str(filtered_index[ii]), fontsize=15)
    plt.text(-375, range_filtered[ii]*16, str(ii), size=25)
    
## mapping filtered MFs
plt.figure(figsize=(12,25))
plt.title(str(specimen) + '_' + str(electrode) + ' using SNR-filter')
plt.imshow(bg[:, :], vmax=brightness)  # avg moco bg
# plotting mask ROIs with numbering
for ii in range(len(activity_filtered)):
    plt.scatter(x_filtered[ii], y_filtered[ii], marker="$" +
                # str(filtered_index[ii])+"$", color='w', linewidths=0.5)
                str(ii)+"$", color='w', linewidths=0.5)

# %% NETWORK ANALYSIS
corr_type = 'pearson'
# choose between pearson, kendall, or spearman
# pearson is used in the study

optimisation_type = 'corr_th_emd'
# choose between corr_th_emd, corr_th_n-comm, or louvain_emd

if optimisation_type == 'corr_th_emd':
    #### corr_th EMD optimisation
    # this is the EMD optimisation discussed in the study
    
    # setting up known distribution for comparison
    distribution_known = []
    for ii in range(len(mu)):
        distribution_known.append(len(mu[ii]))
    
    
    ## correlation matrix
    # using pandas for more varieties of correlation matrix
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    
    corr_mat = activity_filtered_pd.corr(corr_type)

    # convert to nmp for further use
    corr_mat = corr_mat.to_numpy()
    nmp.fill_diagonal(corr_mat, 0)
    
    # loop parameters
    iterations = 2000
    step = 1
    distribution_obs_list = []
    emd_list = []
    
    # optimisation loop
    for ii in range(0,iterations,step):
        corr_th = (ii-1000)/iterations
        corr_mat[corr_mat < corr_th] = 0
        nmp.fill_diagonal(corr_mat, 0)
        
        g = nwx.from_numpy_array(corr_mat)
        c = list(nwx.community.louvain_communities(g, resolution = 0, threshold = 0, seed=2)) # most distinct coms
        ncomm = len(c)
        
        distribution_obs = []
        for ii in range(0, ncomm):
            comm_size = len(c[ii])
            distribution_obs.append(comm_size)
        distribution_obs_list.append(distribution_obs)
        
        # calculate EMD
        emd_list.append(stt.wasserstein_distance(distribution_known, distribution_obs))
    
    optimal_corr_th = (emd_list.index(min(emd_list))-1000)/iterations*step
    
    plt.plot(emd_list)
    plt.title('Optimising correlation threshold using EMD')
    plt.xticks((0,200,400,600,800,1000),(0,0.2,0.4,0.6,0.8,1))
    plt.xlabel('Correlation threshold')
    plt.ylabel("Earth Mover's Distance")
    
    # rerunning with optimal corr_th
    # first need to reset correlation matrix (always removed at end due to max corr_th)
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    corr_mat = activity_filtered_pd.corr(corr_type)
    corr_mat = corr_mat.to_numpy()
    
    # plotting
    plt.figure(figsize = (15,10))
    plt.subplot(121)
    plt.imshow(corr_mat)
    plt.title('Correlation matrix using ' + str(corr_type) + ' coefficient')
    
    # using optimal corr_th
    nmp.fill_diagonal(corr_mat, 0)
    corr_mat[corr_mat < optimal_corr_th] = 0
    plt.subplot(122)
    plt.imshow(corr_mat)
    plt.title('Correlation matrix with ' + str(optimal_corr_th) + ' correlation threshold')
    
    # making the network
    g = nwx.from_numpy_array(corr_mat)
    pos = nwx.spring_layout(g, seed = 12345678)

    
    # cluster algorithm
    c = list(nwx.community.louvain_communities(g, resolution = 0, threshold = 0, seed=2)) # most distinct coms
    ncomm = len(c)
    
    print('Optimal correlation threshold = ' + str(optimal_corr_th))
    print('EMD = ' + str(min(emd_list)))
    print('ncomm = ' + str(ncomm))

if optimisation_type == 'corr_th_ncomm':
    #### corr_th n-comm optimisation
    # this is the n-cluster optimisation discussed in the study
  
    ## correlation matrix
    # using pandas for more varieties of correlation matrix
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    
    corr_mat = activity_filtered_pd.corr(corr_type)

    # convert to nmp for further use
    corr_mat = corr_mat.to_numpy()
    nmp.fill_diagonal(corr_mat, 0)
    
    # loop parameters
    iterations = 1000
    step = 1
    distribution_obs_list = []
    ncomm_list = []
    corr_th_list = []
    
    # optimisation loop
    for ii in range(0,iterations,step):
        corr_th = ii/iterations
        corr_th_list.append(corr_th)
        
        corr_mat[corr_mat < corr_th] = 0
        nmp.fill_diagonal(corr_mat, 0)
        
        g = nwx.from_numpy_array(corr_mat)
        c = list(nwx.community.louvain_communities(g, resolution = 0, threshold = 0, seed=2)) # most distinct coms
        ncomm = len(c)
        ncomm_list.append(ncomm)
    
    plt.figure()
    plt.plot(corr_th_list,ncomm_list)
    plt.xlabel('Correlation threshold')
    plt.ylabel('n communities')
    
    # finding optimal corr_th (closest to n)
    def closest(lst, K):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-K))]
    
    lst = ncomm_list
    K = len(mu)
    
    # saving optimal value for further analysis
    # optimal_corr_th = 1
    optimal_corr_th = (ncomm_list.index(closest(lst, K+1)))/iterations*step
    
    
    # rerunning with optimal corr_th
    # first need to reset correlation matrix (always removed at end due to max corr_th)
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    corr_mat = activity_filtered_pd.corr(corr_type)
    corr_mat = corr_mat.to_numpy()
    
    # plotting
    plt.figure(figsize = (15,10))
    plt.subplot(121)
    plt.imshow(corr_mat)
    plt.title('Correlation matrix using ' + str(corr_type) + ' coefficient')
    
    # using optimal corr_th
    nmp.fill_diagonal(corr_mat, 0)
    corr_mat[corr_mat < optimal_corr_th] = 0
    plt.subplot(122)
    plt.imshow(corr_mat)
    plt.title('Correlation matrix with ' + str(optimal_corr_th) + ' correlation threshold')
    
    # making the network
    g = nwx.from_numpy_array(corr_mat)
    pos = nwx.spring_layout(g, seed = 12345678)
    
    # cluster algorithm
    c = list(nwx.community.louvain_communities(g, resolution = 0, threshold = 0, seed=2)) # most distinct coms
    ncomm = len(c)
    
    print('Optimal correlation threshold = ' + str(optimal_corr_th))
    print('ncomm to reference ratio = ' + str(ncomm/K))
    print('ncomm = ' + str(ncomm))

if optimisation_type == 'louvain_emd':
    #### Louvain EMD optimisation
    # This method optimises using another parameter than the correlation threshold
    # it instead uses EMD to optimise the resolution parameter of the Louvain
    # clustering algorithm used
    # it is not discussed in the study
    
    # setting up known distribution for comparison
    distribution_known = []
    for ii in range(len(mu)):
        distribution_known.append(len(mu[ii]))
    
    
    ## correlation matrix
    # using pandas for more varieties of correlation matrix
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    
    corr_mat = activity_filtered_pd.corr(corr_type)
    # convert to nmp for further use
    corr_mat = corr_mat.to_numpy()
    nmp.fill_diagonal(corr_mat, 0)
    
    # loop parameters
    iterations = 50
    step = 1
    distribution_obs_list = []
    emd_list = []
    
    # optimisation loop
    for ii in range(0, iterations, step):
        g = nwx.from_numpy_array(corr_mat)
        c = list(nwx.community.louvain_communities(g, resolution = ii, seed=2)) # most distinct coms
        ncomm = len(c)
        
        distribution_obs = []
        for ii in range(0, ncomm):
            comm_size = len(c[ii])
            distribution_obs.append(comm_size)
        distribution_obs_list.append(distribution_obs)
        
        # calculate EMD
        emd_list.append(stt.wasserstein_distance(distribution_known, distribution_obs))
    
    optimal_res = emd_list.index(min(emd_list))
    
    
    plt.plot(emd_list)
    plt.title('Optimising Louvain resolution using EMD')
    plt.xlabel('Louvain resolution')
    plt.ylabel("Earth Mover's Distance")
    
    # rerunning with optimal corr_th
    # first need to reset correlation matrix (always removed at end due to max corr_th)
    activity_filtered_pd = pd.DataFrame(activity_filtered)
    activity_filtered_pd = activity_filtered_pd.transpose()
    corr_mat = activity_filtered_pd.corr(corr_type)
    corr_mat = corr_mat.to_numpy()
    nmp.fill_diagonal(corr_mat, 0)
    
    
    # making the network
    g = nwx.from_numpy_array(corr_mat)
    pos = nwx.spring_layout(g, seed = 12345678)

    
    # cluster algorithm
    c = list(nwx.community.louvain_communities(g, resolution = optimal_res, seed=2)) # most distinct coms
    ncomm = len(c)

    print('optimal louvain resolution = ' + str(optimal_res))
    print('EMD = ' + str(min(emd_list)))
    print('ncomm = ' + str(ncomm))

#### GRAPHS

# colour-blind friendly colour cycle
cb_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00',
            '#89cff0', '#000000']*ncomm 

shp = (['o','^','s','v','D','X','p']*ncomm) # varying marker shapes

# plotting abstract network w communities
plt.figure(figsize=(5,5))
plt.title('MFs coloured by clusters (' + str(specimen) + '_' + str(electrode) + ')')
nwx.draw_networkx_edges(g, pos, width=0.1)

# add nodes, using colormap to color according to community number:
for ii in range(ncomm):
    nodelist=list(c[ii])
    nwx.draw_networkx_nodes(g, pos, nodelist = nodelist, node_color = cb_cycle[ii], node_size = 45, node_shape = shp[ii])
    nwx.draw_networkx_labels(g, pos, font_size = 4, font_color = 'w')

#### comparing to matlab distribution
plt.figure(figsize=(15, 12))
plt.subplot(211)
plt.title(str(specimen) + '_' + str(electrode) + ' overlap with matlab MUs using SNR-filter')
plt.imshow(bg[:, :], vmax=brightness)  # avg moco bg
# plotting mask ROIs with numbering
for ii in range(len(activity_filtered)):
    plt.scatter(x_filtered[ii], y_filtered[ii], marker="$" +
                str(filtered_index[ii])+"$", color='w', linewidths=0.5)
for iii in range(len(mu)):
    for ii in mu[iii]:
        plt.plot(x[ii], y[ii], shp[iii], 
                 markerfacecolor = cb_cycle[iii], markeredgecolor = 'w')

plt.subplot(212)
plt.imshow(bg[:,:], vmax = brightness) # avg moco bg
plt.title('(' + str(specimen) + '_' + str(electrode) + ') filtered ROI communities')
for ii in range(0, ncomm):
    nodelist = list(c[ii])
    plt.plot(x_filtered[nodelist], y_filtered[nodelist], shp[ii],
             markerfacecolor = cb_cycle[ii], markeredgecolor = 'w',
               label = 'Community #' + str(ii))
