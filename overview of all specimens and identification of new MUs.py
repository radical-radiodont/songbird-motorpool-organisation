# -*- coding: utf-8 -*-
"""
@author: Frederik Strid
Part of thesis project: ORGANISATION OF THE MOTOR POOL IN BIRDSONG MOTOR CONTROL

This is the main script for testing stimulated and living fibres for all 
specimens and all electrodes. It also contains the code for running the big 
overview figure (one specimen at a time) and for identifying new motor units 
based on activity signals. 

Parts of the script (mainly the calculation of DFF per ROI and some of the 
network analysis) is based on a tutorial from CalTech (Andreev 2023),
which can be found here:
https://focalplane.biologists.com/2023/10/27/analyzing-calcium-imaging-data-using-python/

"""

# %%% PACKAGES
import tifffile
import numpy as nmp # I'm aware np is the usual shorthand but I like my 3-letter abbreviations
from matplotlib import pyplot as plt
from scipy import signal as sig
import pandas as pds
from pointpats import PointPattern
import pointpats.quadrat_statistics as qst
from scipy import spatial as spt

# %% SETTING UP THE DATA
# This section loads in the selected specimen, stitches it together, loads the
# fibre coordinates and the reference motor units

#### specimen selection
specimen = 'gw65' # choose from gw65, gw64, gw55, and gw55_R1 (the active half)

# specimen GW65 is annotated
if specimen == 'gw65':
    #### gw65
    x1 = 195 # x coordinate for edge of overlap for 65r2
    x2 = 22 # x coordinate for edge of overlap for 65r1

    ## moco series
    print('Loading ' + specimen + ' electrodes') 
    
    # PURPLE ELECTRODE
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
    dff_purple = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
    
    # loading in the two halves in their respective parts of the array
    # in a way so that they match up spatially and temporally
    dff_purple[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
    dff_purple[:,:,x1:] = mocodat2[trig2:,:,x2:]
    
    # redefine dimensions
    nt_purple, ny, nx = nmp.shape(dff_purple[:,:,:]) 
    print('Purple...')
    
    # ORANGE ELECTRODE
    # same procedure as for purple
    trig1 = 79
    trig2 = 83
    
    moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series029_ch01_R2_orange_el_moco.tif'
    moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series011_ch01_R1_orange_el_moco.tif'

    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
    
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
    dff_orange = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
    
    dff_orange[:,:,:x1] = mocodat1[trig1:,:,:x1]
    dff_orange[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:,x2:]
    
    nt_orange, ny, nx = nmp.shape(dff_orange[:,:,:]) 
    print('...orange...')
        
    # GREY ELECTRODE
    trig1 = 87
    trig2 = 79
    
    moco1 = 'Data/210303_f_gw65/210303_f_gw65_Series030_ch01_R2_grey_el_moco.tif'
    moco2 = 'Data/210303_f_gw65/210303_f_gw65_Series013_ch01_R1_grey_el_moco.tif'

    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
    
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
    dff_grey = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
    
    dff_grey[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
    dff_grey[:,:,x1:] = mocodat2[trig2:,:,x2:]
    
    nt_grey, ny, nx = nmp.shape(dff_grey[:,:,:])
    print('... and grey')

    # unusued electrodes
    # no data acquired for the green electrode, but for making the next section
    # easier to run for all specimen, an empty array is generated
    dff_green = nmp.zeros_like(dff_purple)
    nt_green, ny, nx = nmp.shape(dff_green[:,:,:]) 

    ## muscle stimulation
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
    
    nt_m, ny_m, nx_m = nmp.shape(muscle_moco[:,:,:]) 

    ## averaged dff data, used as a background in many plots
    print('Loading ' + specimen + ' average image data')
    avg1 = 'Data/210303_f_gw65/AVG_210303_f_gw65_MSD_R_Series032_ch01_R2 for_moco.tif'
    avg2 = 'Data/210303_f_gw65/AVG_210303_f_gw65_MSD_R_Series017_ch01_R1_for moco.tif'
    
    avgdat1 = tifffile.imread(avg1)
    avgdat2 = tifffile.imread(avg2)
    
    ny1, nx1 = nmp.shape(avgdat1[:,:]) 
    ny2, nx2 = nmp.shape(avgdat2[:,:]) 
    
    avg_moco = nmp.zeros((ny1, (nx1+nx2)-(x2+(200-x1))))
    avg_moco[:,:x1] = avgdat1[:,:x1]
    avg_moco[:,x1:] = avgdat2[:,x2:]
    
    bg = avg_moco
    
    # brightness for the background
    brightness = 500
    
    
    ## MF coordinates
    # loaded in for each half and will be assembled later
    print('Loading ' + specimen + ' mask data') 
    mask_r1 = pds.read_csv('Data/210303_f_gw65/210303_f_gw65_MSD_R_Series017_RAW_ch01_mask_for_python.csv')
    mask_r2 = pds.read_csv('Data/210303_f_gw65/210303_f_gw65_MSD_R_Series032_RAW_ch01_mask_for_python.csv')

    ## known mus
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

elif specimen == 'gw64': 
    #### gw64
    # for more detailed annotation see GW64
    x1 = 198
    x2 = 40
    
    ## GREY ELECTRODE  
    print('Loading ' + specimen + ' electrodes')
    trig1 = 57
    trig2 = 21
    
    moco1 = 'Data/210305_f_gw64/210305_f_gw64_Series028_ch01_grey_el_moco_fixed.tif'
    moco2 = 'Data/210305_f_gw64/210305_f_gw64_Series034_ch01_R2_grey_el_moco.tif'
    
    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
    
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
    
    dff_grey = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
    
    dff_grey[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
    dff_grey[:,:,x1:] = mocodat2[trig2:,:,x2:]
    
    nt_grey, ny, nx = nmp.shape(dff_grey[:,:,:]) 
    print("Grey. That's it.")
    
    
    ## unused electrodes
    dff_purple = nmp.zeros_like(dff_grey)
    nt_purple, ny, nx = nmp.shape(dff_purple[:,:,:]) 
    dff_orange = nmp.zeros_like(dff_grey)
    nt_orange, ny, nx = nmp.shape(dff_orange[:,:,:]) 
    dff_green = nmp.zeros_like(dff_grey)
    nt_green, ny, nx = nmp.shape(dff_green[:,:,:]) 
    
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
    print('Loading ' + specimen + ' average image data')
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
    brightness = 1300
    
    
    ## mask
    print('Loading ' + specimen + ' mask data')
    mask_r1 = pds.read_csv('Data/210305_f_gw64/210305_f_gw64_MDS_L_Series037_RAW_ch01_mask_for_python.csv')
    mask_r2 = pds.read_csv('Data/210305_f_gw64/210305_f_gw64_MDS_L_Series030_RAW_ch01_mask_for_python.csv')
    
    
    ## known mus
    mu_1 = [128,159,116,81,100,102]
    mu_2 = [101]
    mu_3 = [95]
    mu_4 = [68,73,74,75,76]
    
    mu = [mu_1,mu_2,mu_3,mu_4]
    mu_list = mu_1+mu_2+mu_3+mu_4

elif specimen == 'gw55':
    #### gw55
    # this is the whole of the GW55 but due to issues in the original data, ony
    # one half is good for producing activity signals, making it functionally
    # identical to running only that half (see next specimen) but showing the
    # whole for a better idea of the MU distribution
    x1 = 123
    x2 = 0
    
    ## moco series
    print('Loading ' + specimen + ' electrodes')
        
    # PURPLE ELECTRODE
    # only DFF data from the functional half is used
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series006_ch01_purple_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_purple = mocodat1
    
    nt_purple, ny, nx = nmp.shape(dff_purple[:,:,:]) 
    print('Purple...')

    # PURPLE ELECTRODE, WHOLE (FOR SCAN IMG)
    # only used for visual reference    
    trig1 = 46
    trig2 = 46
    
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series006_ch01_purple_el_moco.tif'
    moco2 = 'Data/210304_f_gw55/210304_f_gw55_Series030_ch01_R2_purple_el_moco.tif'

    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
    
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 

    dff_purple_whole = nmp.zeros((nt2-trig2, ny1-16, (nx1+nx2)-(x2+(200-x1))))
    
    dff_purple_whole[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:ny1-16,:x1]
    dff_purple_whole[:,:,x1:] = mocodat2[trig2:,16:,x2:]


    # ORANGE ELECTRODE
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series014_ch01_orange_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_orange = mocodat1
    
    nt_orange, ny, nx = nmp.shape(dff_orange[:,:,:]) 
    print('...orange...')
    
    # GREEN ELECTRODE
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series009_ch01_green_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_green = mocodat1
    
    nt_green, ny, nx = nmp.shape(dff_green[:,:,:]) 
    print('... and green')

    # unused electrodes
    dff_grey = nmp.zeros_like(dff_purple)
    nt_grey, ny, nx = nmp.shape(dff_grey[:,:,:]) 
    
    ## muscle stim
    # also only data from the functional half
    muscle_moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series019_ch01_muscle_moco.tif'
    muscle_dat1 = tifffile.imread(muscle_moco1)
    
    muscle_moco = muscle_dat1
    

    ## avg image
    # shows the whole specimen
    print('Loading ' + specimen + 'average image data')
    avg1 = 'Data/210304_f_gw55/AVG_210304_f_gw55_MDS_R_Series021_ch01_for_moco.tif'
    avg2 = 'Data/210304_f_gw55/AVG_210304_f_gw55_MDS_R_Series029_ch01_R2_for_moco.tif'
    
    avgdat1 = tifffile.imread(avg1)
    avgdat2 = tifffile.imread(avg2)
    
    ny1, nx1 = nmp.shape(avgdat1[:,:]) 
    ny2, nx2 = nmp.shape(avgdat2[:,:])
    
    avg_moco = nmp.zeros((ny1-16, (nx1+nx2)-(x2+(200-x1))))
    avg_moco[:,:x1] = avgdat1[:ny1-16,:x1]
    avg_moco[:,x1:] = avgdat2[16:,x2:]
    
    ## brightness for bg
    brightness = 1000

    
    ## mask
    print('Loading ' + specimen + ' mask data')
    mask_r1 = pds.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series029_RAW_ch01_mask_for_python.csv')
    mask_r2 = pds.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series021_RAW_ch01_mask_for_python.csv')
    
    ## known mus
    mu_1 = [31]
    mu_2 = [66]
    mu_3 = [24,25,72,14]
    mu_4 = [3,53,62]
    mu_5 = [33,52,41]
    mu_6 = [58]
    mu_7 = [51,71]
    
    mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6,mu_7]
    mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7

elif specimen == 'gw55_R1': 
    #### gw55_R1
    # this is ONLY the functional half, with no visualisation of the whole 
    # specimen. Does not put the MUs in proper context but does give a more
    # accurate look at what happens behind the scenes
    x1 = 123
    x2 = 0
    
    ## moco series
    print('Loading ' + specimen + ' electrodes')
        
    # PURPLE ELECTRODE
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series006_ch01_purple_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_purple = mocodat1
    
    nt_purple, ny, nx = nmp.shape(dff_purple[:,:,:]) 
    print('Purple...')

    # ORANGE ELECTRODE
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series014_ch01_orange_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_orange = mocodat1
    
    nt_orange, ny, nx = nmp.shape(dff_orange[:,:,:]) 
    print('...orange...')
    
    # GREEN ELECTRODE
    moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series009_ch01_green_el_moco.tif'
    mocodat1 = tifffile.imread(moco1)
    
    dff_green = mocodat1
    
    nt_green, ny, nx = nmp.shape(dff_green[:,:,:]) 
    print('... and green')

    # unused electrodes
    dff_grey = nmp.zeros_like(dff_purple)
    nt_grey, ny, nx = nmp.shape(dff_grey[:,:,:]) 
    
    
    ## muscle stim
    muscle_moco1 = 'Data/210304_f_gw55/210304_f_gw55_Series019_ch01_muscle_moco.tif'
    muscle_dat1 = tifffile.imread(muscle_moco1)
    
    muscle_moco = muscle_dat1
    

    ## avg image
    print('Loading ' + specimen + '_ average image data')
    avg1 = 'Data/210304_f_gw55/AVG_210304_f_gw55_MDS_R_Series021_ch01_for_moco.tif'

    avgdat1 = tifffile.imread(avg1)
    ny1, nx1 = nmp.shape(avgdat1[:,:]) 

    avg_moco = avgdat1
        
    ## brightness for bg
    brightness = 1000

    
    ## mask
    print('Loading ' + specimen + ' mask data')
    mask_r1 = pds.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series029_RAW_ch01_mask_for_python.csv')
    mask_r2 = pds.read_csv('Data/210304_f_gw55/210304_f_gw55_MDS_R_Series021_RAW_ch01_mask_for_python.csv')
    
    
    ## known mus
    mu_1 = [31]
    mu_2 = [66]
    mu_3 = [24,25,72,14]
    mu_4 = [3,53,62]
    mu_5 = [33,52,41]
    mu_6 = [58]
    mu_7 = [51,71]
    
    mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6,mu_7]
    mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7

elif specimen == '7391':
    #### 7391
    x1 = 200
    x2 = 45
     
    ## moco series    
    print('Loading ' + specimen + ' MOCO data')
    
    ## PURPLE ELECTRODE
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
   
    dff_purple = nmp.zeros((nt2-trig2, ny1, (nx1+nx2)-(x2+(200-x1))))
   
    dff_purple[:,:,:x1] = mocodat1[trig1:nt2-trig2+trig1,:,:x1]
    dff_purple[:,:,x1:] = mocodat2[trig2:,:ny2-32,x2:]

    nt_purple, ny, nx = nmp.shape(dff_purple[:,:,:]) 
    print('Purple...')
   
   
    ## GREY ELECTRODE
    trig1 = 53
    trig2 = 117
    
    moco1 = 'Data/210303_f_7391/210303_f_7391_Series045_ch01_R2_grey_el_moco.tif'   
    moco2 = 'Data/210303_f_7391/210303_f_7391_Series030_ch01_R1_grey_el_moco.tif'   
    
    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
    
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 

    mocodat2 = nmp.append(nmp.zeros((nt2,32,nx2)),mocodat2,axis=1)
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
   
    dff_grey = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
    
    dff_grey[:,:,:x1] = mocodat1[trig1:,:,:x1]
    dff_grey[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:ny2-32,x2:]
    
    nt_grey, ny, nx = nmp.shape(dff_grey[:,:,:]) 
    print('...grey...')
       

    # GREEN ELECTRODE
    trig1 = 66
    trig2 = 118
   
    moco1 = 'Data/210303_f_7391/210303_f_7391_Series047_ch01_R2_green_el_moco.tif'   
    moco2 = 'Data/210303_f_7391/210303_f_7391_Series025_ch01_R1_green_el_moco.tif'   
   
    mocodat1 = tifffile.imread(moco1)
    mocodat2 = tifffile.imread(moco2)
   
    nt1, ny1, nx1 = nmp.shape(mocodat1[:,:,:]) 
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 

    mocodat2 = nmp.append(nmp.zeros((nt2,32,nx2)),mocodat2,axis=1)
    nt2, ny2, nx2 = nmp.shape(mocodat2[:,:,:]) 
   
    dff_green = nmp.zeros((nt1-trig1, ny1, (nx1+nx2)-(x2+(200-x1))))
   
    dff_green[:,:,:x1] = mocodat1[trig1:,:,:x1]
    dff_green[:,:,x1:] = mocodat2[trig2:nt1-trig1+trig2,:ny2-32,x2:]
   
    nt_green, ny, nx = nmp.shape(dff_green[:,:,:]) 
    print('...and green')
   
   
    ## unusued electrodes
    dff_orange = nmp.zeros_like(dff_grey)
    nt_orange, ny, nx = nmp.shape(dff_orange[:,:,:]) 


    ## avg image
    print('Creating ' + specimen + ' average image data')
    avg_moco = nmp.mean(dff_purple, axis = 0)
   
    ## brightness for bg
    brightness = 600


    ## mask
    print('Loading ' + specimen + ' mask data')
    mask_r1 = pds.read_csv('Data/210303_f_7391/R1_210303_f_7391_MSD_R_Series0XX_RAW_ch01_mask_for_python.csv')
    mask_r2 = pds.read_csv('Data/210303_f_7391/R2_210303_f_7391_MSD_R_Series0XX_RAW_ch01_mask_for_python.csv')


   
    ## known MUs
    mu_1 = [14,121]
    mu_3 = [64,92,97,98,106,110,179,187]
    mu_4 = [119,112,117,13,128,190]
    mu_5 = [16,17,198]
    mu_6 = [151,103,94]
    mu_7 = [206]
    mu_8 = [130]
    mu_9 = [136]
    mu_10 = [152]
    mu_11 = [133,153]
    mu_12 = [132]
    mu_13 = [35]
    mu_14 = [47]
    mu_15 = [162,163,193,199]
    mu_16 = [73,74]
    mu_17 = [53,63,77]
    mu_18 = [11,188]
    mu_19 = [109,79,19]
   
    mu = [mu_1,mu_2,mu_3,mu_4,mu_5,mu_6,mu_7,mu_8,mu_9,mu_10,mu_11,mu_12,mu_13,mu_14,mu_15,mu_16,mu_17,mu_18,mu_19]
    mu_list = mu_1+mu_2+mu_3+mu_4+mu_5+mu_6+mu_7+mu_8+mu_9+mu_10+mu_11+mu_12+mu_13+mu_14+mu_15+mu_16+mu_17+mu_18+mu_19

else:
    print('Invalid specimen identifier')
    
# for easy looping
electrodes = [dff_purple,dff_orange,dff_grey,dff_green]
electrode_names = ['purple','orange','grey','green']
nt = [nt_purple,nt_orange,nt_grey,nt_green]

if specimen != '7391':
    # specimen 7391 does not have a functional MOCO file for field-stimulation
    # and can therefore not have live-fibre detection
    nt_m, ny_m, nx_m = nmp.shape(muscle_moco[:,:,:])

print('Specimen ' + specimen + ' loaded')

# %%% PROCESSING THE DATA
# this section deals with stitching the mask, finding the living and stimulated 
# fibres (per electrode), calculates the actual DFF, and produces a big overview
# plot of these parameters for the given specimen


#### mask coords
print('Generating mask')

# extracting individual XY coords from the mask
x_r1,y_r1 = mask_r1.get('x'), mask_r1.get('y')
x_r2,y_r2 = mask_r2.get('x'), mask_r2.get('y')

# removing overlapping XY coords
if specimen == 'gw55_R1' or specimen == 'gw55' or specimen == '7391':
    # because gw55 is stitched right on left not left on right
    for ii in range(0, len(x_r1)): 
        if x_r1[ii] < nx1-x1:
            x_r1[ii] = x_r1[ii]+nx     
else:
    for ii in range(0, len(x_r1)):
        if x_r1[ii] < x2:
            x_r1[ii] = x_r1[ii]+nx 
            # moved out of frame and removed when later removing those
            # close to the edge

# shifting coords right to align with r1
x_r1 = x_r1+x1-x2 

# shifting the vertical alignment for gw55 and 7391
# because for both specimen the original scans do not align in the Y-plane
if specimen == 'gw55_R1' or specimen == 'gw55':
    y_r1 = y_r1-16
if specimen == '7391':
    y_r1 = y_r1+32

# making a combined list of coordinates now that both sides are aligned
# and overlap has been removed
x = pds.concat([x_r2, x_r1], ignore_index = True)
y = pds.concat([y_r2, y_r1], ignore_index = True)

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

## plotting the mask
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

## mu coordinates
# later procedures will read y coords wrong unless flipped first
flip_y = []
for ii in y:
    flip_y.append(-abs(ii))
    
flip_y = pds.Series(flip_y)

# merging the lists of x and y coords into a [x,y] format
def merge(list1, list2):
 
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
     
    return merged_list

mu_coords = merge(x,flip_y)


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

#### calculating DFF
print('Calculating DFF')
# this is the dF/F calculation for the electrode data, which will later be used
# to stimulated fibres
# it follows the same basic procedure as the DFF for the field-stimulation except
# now it's done for all of the electrodes
# (this is usually the most taxing part of this script)

## pre-initialising the DFF per ROI for each electrode
activity_purple = nmp.zeros((len(alive), nt_purple))
activity_orange = nmp.zeros((len(alive), nt_orange))
activity_grey = nmp.zeros((len(alive), nt_grey))
activity_green = nmp.zeros((len(alive), nt_green))

activity_electrodes = [activity_purple,activity_orange,activity_grey,activity_green]

# looping through each of the electrodes
for jj in range(len(electrodes)):
    # setting the current electrode of the loop
    electrode = electrodes[jj]
    # for saving the data in the loop
    activity_electrode = activity_electrodes[jj]
    
    dff = nmp.zeros_like(electrode)
    F0 = nmp.mean(electrode[:,:,:], axis = (0))
    
    for ii in range(nt[jj]):
        dff[ii,:,:] = nmp.array((electrode[ii,:,:]-F0)/F0)
    
    # same as done for the field-stimulation
    for ii in range(len(alive)):
        xy_roi_dff = dff[:,y_alive[ii] + nmp.arange(-w,w),:][:,:,x_alive[ii] + nmp.arange(-w,w)]
        xy_roi_dff_mean = nmp.mean(xy_roi_dff,(1,2))
        activity_electrode[ii,:] = xy_roi_dff_mean
    
    activity_electrode = activity_electrode[~nmp.isnan(activity_electrode).any(axis = 1)]    
    activity_electrodes[jj] = activity_electrode
    print(electrode_names[jj] + ' electrode done')
    # moving on to next electrode
    

#### detecting stimulated fibres
print('Detecting active signals')

# pre-initialising arrays and lists for the signals and coords of the stimulated
# fibres, again per electrode
activity_purple_filtered = nmp.zeros_like(activity_purple)
x_purple_filtered = []
y_purple_filtered = []
filtered_index_purple = []
activity_orange_filtered = nmp.zeros_like(activity_orange)
x_orange_filtered = []
y_orange_filtered = []
filtered_index_orange = []
activity_grey_filtered = nmp.zeros_like(activity_grey)
x_grey_filtered = []
y_grey_filtered = []
filtered_index_grey = []
activity_green_filtered = nmp.zeros_like(activity_green)
x_green_filtered = []
y_green_filtered = []
filtered_index_green = []
    
activity_filtered_electrodes =[activity_purple_filtered,activity_orange_filtered,activity_grey_filtered,activity_green_filtered]
x_filtered_electrodes = [x_purple_filtered,x_orange_filtered,x_grey_filtered,x_green_filtered]
y_filtered_electrodes = [y_purple_filtered,y_orange_filtered,y_grey_filtered,y_green_filtered]
filtered_index_electrodes = [filtered_index_purple,filtered_index_orange,filtered_index_grey,filtered_index_green]
    
# the signal quality is varied between the specimens so the threshold for
# stimulation has to be adjusted
if specimen == 'gw65':
    sd_mult = 5.5
elif specimen == 'gw64':
    sd_mult = 5
elif specimen == 'gw55_R1' or specimen == 'gw55':
    sd_mult = 4.5
elif specimen == '7391':
    sd_mult = 8

# again looping through each of the electrodes
for jj in range(len(electrodes)):
    activity = activity_electrodes[jj]
    activity_filtered = activity_filtered_electrodes[jj]
    
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
            x_filtered_electrodes[jj].append(x_alive[ii])
            y_filtered_electrodes[jj].append(y_alive[ii])
            filtered_index_electrodes[jj].append(ii)

# converting coords to pandas series and removing the empty rows left over
# in the activity arrays (corresponding to those fibres that did not have
# any significant activity in them)
x_purple_filtered = pds.Series(x_purple_filtered)
y_purple_filtered = pds.Series(y_purple_filtered)
activity_purple_filtered = activity_purple_filtered[~nmp.all(activity_purple_filtered == 0, axis=1)]
x_orange_filtered = pds.Series(x_orange_filtered)
y_orange_filtered = pds.Series(y_orange_filtered)
activity_orange_filtered = activity_orange_filtered[~nmp.all(activity_orange_filtered == 0, axis=1)]
x_grey_filtered = pds.Series(x_grey_filtered)
y_grey_filtered = pds.Series(y_grey_filtered)
activity_grey_filtered = activity_grey_filtered[~nmp.all(activity_grey_filtered == 0, axis=1)]
x_green_filtered = pds.Series(x_green_filtered)
y_green_filtered = pds.Series(y_green_filtered)
activity_green_filtered = activity_green_filtered[~nmp.all(activity_green_filtered == 0, axis=1)]

# n of stimulated fibres per electrode, reported later in the plots
n_actives = [len(x_purple_filtered),len(x_orange_filtered),len(x_grey_filtered),len(x_green_filtered)]

#### grand overview plot
# plot visualising all of the above as well as some more analysis on MUs
print('Plotting plots')

if specimen == 'gw55_R1':
    # because this is half the size of the other specimens
    font = 6
else:
    font = 8
    
plt.figure(figsize=(30,15))

# list of colours to be iterated through in MU plots
# (colour-blind friendly colours used for the author's own sanity)
cb_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00',
            '#89cff0', '#000000']*len(mu)

# list of marker shapes to be iterated through in MU plots
# for those were 'colour-blind friendly' doesn't cut it
shp = (['o','^','s','v','D','X','p']*len(mu))

#### raw image
# shows the first frame of the raw scan for comparison with the average
plt.subplot2grid((6,2),(0,0), colspan = 2)
plt.title('specimen = ' + specimen)
plt.xticks(ticks = ())
plt.yticks(ticks = ())

if specimen == 'gw64':
    plt.imshow(dff_grey[0,:,:], vmax = brightness) # avg moco bg    
elif specimen == 'gw55':
    # here we use the whole scan
    plt.imshow(dff_purple_whole[0,:,:], vmax = brightness) # avg moco bg   
else:
    plt.imshow(dff_purple[0,:,:], vmax = brightness) # avg moco bg


#### mask
# shows the location of ALL fibres in the specimen
plt.subplot2grid((6,2),(1,0), colspan = 2)
plt.text(5,10, 'n = ' + str(nroi), color = 'w', size = font)
plt.xticks(ticks = ())
plt.yticks(ticks = ())

bg = avg_moco # using the average scan as the background for the rest of the plots
plt.imshow(bg[:,:], vmax = brightness)

for i in range(0,len(x)):
    plt.plot(x[i],y[i], 'o', mfc = 'none', mec = 'w', markersize = 5)
    # alternative marker showing the fibre ID instead of just circles
    # plt.scatter(x[i],y[i], marker="$"+str(i)+"$", color = 'w', linewidths = 0.5) 

#### dead or alive
if specimen != '7391':
    # again, specimen 7391 lacks the field-stim MOCO for detecting alive fibres
    
    # shows which fibres are registered as dead or alive
    plt.subplot2grid((6,2),(2,0), colspan = 2)
    plt.text(5,10, 'n alive = ' + str(len(alive)) + ', n dead = ' + str(len(dead)), color = 'w', size = font)
    plt.xticks(ticks = ())
    plt.yticks(ticks = ())
    if specimen == 'gw65':
        plt.imshow(muscle_moco[583,:,])
    elif specimen == 'gw64':
        plt.imshow(muscle_moco[313,:,:])
    elif specimen == 'gw55_R1' or specimen == 'gw55':
        plt.imshow(muscle_moco[552,:,:])
    
    for i in range(len(alive)):
        plt.plot(x_alive[i],y_alive[i], 'o', mfc = 'none', mec = 'w', markersize = 5)
        # alternative marker showing the fibre ID instead of just circles
        # plt.scatter(x_alive[i],y_alive[i], marker="$"+str(i)+"$", color = 'w', linewidths = 0.5)
    for i in range(len(dead)):
        plt.plot(x_dead[i],y_dead[i], 'o', mfc = 'none', mec = 'r', markersize = 5)
        # alternative marker showing the fibre ID instead of just circles
        # plt.scatter(x_dead[i],y_dead[i], marker="$"+str(i)+"$", color = 'r', linewidths = 0.5)


#### electrode coverage for stimulated fibres
# shows the stimulated fibres colour-coded by which electrode they respond to
# (white responds to all relevant electrodes)
plt.subplot2grid((6,2),(3,0), colspan = 2)
plt.text(5,10, str(electrode_names[0]) + ' n mf = ' + str(n_actives[0]), color = 'w', size = font)
plt.text(5,17, str(electrode_names[1]) + ' n mf = ' + str(n_actives[1]), color = 'w', size = font)
plt.text(5,24, str(electrode_names[2]) + ' n mf = ' + str(n_actives[2]), color = 'w', size = font)
plt.text(5,31, str(electrode_names[3]) + ' n mf = ' + str(n_actives[3]), color = 'w', size = font)

plt.xticks(ticks = ())
plt.yticks(ticks = ())
plt.imshow(bg[:, :], vmax=brightness)

# there's a whole lot of combinations to go through here
# only in purple
for ii in range(len(activity_purple_filtered)):
    if filtered_index_purple[ii] not in filtered_index_orange and filtered_index_purple[ii] not in filtered_index_grey and filtered_index_purple[ii] not in filtered_index_green:
        # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'm', markeredgecolor = 'w', markersize = 5)
        
# only in orange
for ii in range(len(activity_orange_filtered)):
    if filtered_index_orange[ii] not in filtered_index_purple and filtered_index_orange[ii] not in filtered_index_grey and filtered_index_orange[ii] not in filtered_index_green:
        # plt.scatter(x_orange_filtered[ii], y_orange_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_orange_filtered[ii], y_orange_filtered[ii], 'o', markerfacecolor = 'C1', markeredgecolor = 'w', markersize = 5)
        
# only in grey
for ii in range(len(activity_grey_filtered)):
    if filtered_index_grey[ii] not in filtered_index_purple and filtered_index_grey[ii] not in filtered_index_orange and filtered_index_grey[ii] not in filtered_index_green:
        # plt.scatter(x_grey_filtered[ii], y_grey_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_grey_filtered[ii], y_grey_filtered[ii], 'o', markerfacecolor = 'C7', markeredgecolor = 'w', markersize = 5)
        
# only in green
for ii in range(len(activity_green_filtered)):
    if filtered_index_green[ii] not in filtered_index_purple and filtered_index_green[ii] not in filtered_index_orange and filtered_index_green[ii] not in filtered_index_grey:
        # plt.scatter(x_green_filtered[ii], y_green_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_green_filtered[ii], y_green_filtered[ii], 'o', markerfacecolor = 'g', markeredgecolor = 'w', markersize = 5)

# present in purple and orange
for ii in range(len(activity_purple_filtered)):
    if filtered_index_purple[ii] in filtered_index_orange and filtered_index_purple[ii] not in filtered_index_grey and filtered_index_purple[ii] not in filtered_index_green:
        # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'm', markerfacecoloralt = 'C1', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)
        
# present in purple and grey
for ii in range(len(activity_purple_filtered)):
    if filtered_index_purple[ii] in filtered_index_grey and filtered_index_purple[ii] not in filtered_index_orange and filtered_index_purple[ii] not in filtered_index_green:
        # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'C7', markerfacecoloralt = 'm', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)
 
# present in purple and green
for ii in range(len(activity_purple_filtered)):
    if filtered_index_purple[ii] in filtered_index_green and filtered_index_purple[ii] not in filtered_index_orange and filtered_index_purple[ii] not in filtered_index_grey:
        # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'm', markerfacecoloralt = 'g', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)

# present in orange and grey
for ii in range(len(activity_orange_filtered)):
    if filtered_index_orange[ii] in filtered_index_grey and filtered_index_orange[ii] not in filtered_index_purple and filtered_index_orange[ii] not in filtered_index_green:
        # plt.scatter(x_orange_filtered[ii], y_orange_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_orange_filtered[ii], y_orange_filtered[ii], 'o', markerfacecolor = 'C1', markerfacecoloralt = 'g', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)

# present in orange and green
for ii in range(len(activity_orange_filtered)):
    if filtered_index_orange[ii] in filtered_index_green and filtered_index_orange[ii] not in filtered_index_purple and filtered_index_orange[ii] not in filtered_index_grey:
        # plt.scatter(x_orange_filtered[ii], y_orange_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_orange_filtered[ii], y_orange_filtered[ii], 'o', markerfacecolor = 'C1', markerfacecoloralt = 'g', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)
 
# present in grey and green
for ii in range(len(activity_grey_filtered)):
    if filtered_index_grey[ii] in filtered_index_green and filtered_index_grey[ii] not in filtered_index_purple and filtered_index_grey[ii] not in filtered_index_orange:
        # plt.scatter(x_grey_filtered[ii], y_grey_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
        plt.plot(x_grey_filtered[ii], y_grey_filtered[ii], 'o', markerfacecolor = 'g', markerfacecoloralt = 'C7', fillstyle = 'left', markeredgecolor = 'w', markersize = 5)
 
# present in all
if specimen == 'gw65':
    for ii in range(len(activity_purple_filtered)):
        if filtered_index_purple[ii] in filtered_index_orange and filtered_index_purple[ii] in filtered_index_grey:
            # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
            plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'w', markeredgecolor = 'w', markersize = 5)


# GW64 only has one electrode

elif specimen == 'gw55_R1' or specimen == 'gw55':
    for ii in range(len(activity_purple_filtered)):
        if filtered_index_purple[ii] in filtered_index_orange and filtered_index_purple[ii] in filtered_index_green:
            # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
            plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'w', markeredgecolor = 'w', markersize = 5)

elif specimen == '7391':
    for ii in range(len(activity_purple_filtered)):
        if filtered_index_purple[ii] in filtered_index_grey and filtered_index_purple[ii] in filtered_index_green:
            # plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" + str(ii)+"$", color='w', linewidths=0.5)
            plt.plot(x_purple_filtered[ii], y_purple_filtered[ii], 'o', markerfacecolor = 'w', markeredgecolor = 'w', markersize = 5)


#### reference motor units
# shows the mapping of the manually identified motor units
plt.subplot2grid((6,2),(4,0), colspan = 2)
plt.imshow(bg[:,:], vmax = brightness)
plt.text(5,10,'n MFs = ' + str(len(mu_list)), color = 'white', size = font)
plt.text(5,17,'n mus = ' + str(len(mu)), color = 'white', size = font)
plt.xticks(ticks = ())
plt.yticks(ticks = ())

# this is where the shape and colour lists become relevant
for iii in range(len(mu)):
    for ii in mu[iii]:
        plt.plot(x[ii], y[ii], shp[iii], 
                 markerfacecolor = cb_cycle[iii], markeredgecolor = 'w')
        
#### motor unit territories and convex hulls
# shows the territory covered by each motor unit (>2 fibres) as a convex hull
# this also includes the quadrat analysis in order to show significance
plt.subplot2grid((6,2),(5,0), colspan = 2)
plt.imshow(bg[:,:], vmax = brightness) # avg moco bg
plt.text(5,10,'n MFs = ' + str(len(mu_list)), color = 'w', size = font)
plt.text(5,17,'n mus = ' + str(len(mu)), color = 'w', size = font)
plt.xticks(ticks = ())
plt.yticks(ticks = ())

µm_per_pix = 1.168 # convert pixels to µm
hull_area = [] # save area of convex hulls for printing

# in this loop, both quadrat analysis and convex hulls are done per MU
for jj in range(len(mu)):
    # saving coordinates of eligible MUs
    if len(mu[jj]) >= 3: # minimum size of mu for both QA and CH
        territory_coords = []
        for ii in mu[jj]:
            territory_coords.append(mu_coords[ii])
        
        # quadrat analysis
        # for an indepth guide on quadrat analysis in python, see:
        # https://pysal.org/notebooks/explore/pointpats/Quadrat_statistics.html
        pp_iii = PointPattern(territory_coords)
        window_x = pp_iii.mbb[2]-pp_iii.mbb[0] # saving the height of the window
        window_y = pp_iii.mbb[3]-pp_iii.mbb[1] # and the width
        # using the window dimensions to make sure that the quadrats are
        # somewhat similar size for all MUs instead of scaling with 
        # size of the territory
        q_r = qst.QStatistic(pp_iii, shape = 'rectangle', nx = round(window_x/(nx/15)+.5), ny = round(window_y/(ny/15)+.5)) # number and type of quadrats

        # convex hull
        territory_coords = nmp.asarray(territory_coords, dtype=nmp.float32)
        territory = spt.ConvexHull(territory_coords)
        
        for simplex in territory.simplices:
            plt.fill(list(territory_coords[simplex, 0]), list(abs(territory_coords[simplex, 1])), edgecolor = cb_cycle[jj], linewidth = 3)
            
        # this is so that later printed text is printed from a point
        # where it does not overlap too much with other MUs
        if specimen == 'gw65' or specimen == 'gw64':
            territory_start = territory_coords[2]
        if specimen == 'gw55' or specimen == '7391':
            territory_start = territory_coords[0]
        # also printing the result of the QA
        plt.text(territory_start[0]-5, abs(territory_start[1])-13, 'p = ' + str("%.3f" % q_r.chi2_pvalue), color = 'w', size = font)

        hull_area.append(territory.volume)
        # and the size of the territory
        plt.text(territory_start[0]-8, abs(territory_start[1])-5, 'Area = ' + str("%.0f" % (territory.volume)) + ' ${px}$', color = 'w', size = font)
        
plt.tight_layout()

# %% IDENTIFYING NEW MOTOR UNITS
# this section expands on the set of reference MUs identified by Adam et al. 2021
# by looking for additional MUs hidden in the activity signals

#### finding stimulated unknown fibres
# new MUs must belong to the subset of stimulated fibres to be considered valid
# all the other activity signals are likely just noise or bleeding

# mapping filtered MFs against reference as a way of seeing which are not
# in the overlap (i.e., the 'unknown')
# note this has to be manually done per electrode (i.e., replacing 'purple' with
# other electrode identifiers, both here and below)
plt.figure(figsize=(12,25))
plt.title(str(specimen) + '_purple', size = 40)
plt.imshow(bg[:, :], vmax=brightness)  # avg moco bg
plt.xticks(ticks = ())
plt.yticks(ticks = ())

# plotting mask ROIs with numbering
for ii in range(len(activity_purple_filtered)):
    plt.scatter(x_purple_filtered[ii], y_purple_filtered[ii], marker="$" +
                # str(filtered_index[ii])+"$", color='w', linewidths=0.5)
                str(ii)+"$", color='w', linewidths=0.5)
for iii in range(len(mu)):
    for ii in mu[iii]:
        plt.plot(x[ii], y[ii], shp[iii], 
                 markerfacecolor = cb_cycle[iii], markeredgecolor = 'w', alpha = 0.45)

# saving the uknown fibres as lists for each specimen
if specimen == 'gw65':
    purple_unknown = [2,5,6,9]
    orange_unknown = [4,8]
    grey_unknown = [4,9,11,12,13,20,23]
    
if specimen == 'gw64':
    grey_unknown = [0,1,2,3,4,5]

if specimen == 'gw55_R1':
    purple_unknown = [0,4,5,6,7]
    orange_unknown = [4,5]
    green_unknown = [0,4]

if specimen == '7391':
    # first batch
    # purple_unknown = [4,11,9,10,3,5,12,8,7,6,2,14,24,26,21,27,19,20,18,22]
    # grey_unknown = [4,3,5,1,0,16,14,12]
    # green_unknown = [2,13,14,6,7,3,5,8,9,4,1,12,23,21,19]
    # second batch (after correcting some misplaced coordinates, does not
    # include those assigned to MUs after the first batch)
    purple_unknown = [4,11,3,7,6,2,25,31,17,23,26,20]
    grey_unknown = [4,5,0,17,16,12]
    green_unknown = [2,14,6,3,8,9,4,12,25,24,22,20]
    
    
# %%% plotting unknown signals
# plotting the activity signals for qualitative comparison of similarity
# shows all electrodes side-by-side, although signals should not be compared
# between electrodes
# note: make sure to silence (conver to comment) the electrodes not present
# in the current specimen (e.g., green for specimen GW65)
# note 2: the gaps are from known fibres they are irrelevant
plt.figure(figsize=(60, 40))


#### purple
activity_purple_filtered_norm = nmp.zeros_like(activity_purple_filtered)
for ii in range(len(activity_purple_filtered)):
    activity_purple_filtered_norm[ii] = (activity_purple_filtered[ii] - nmp.min(activity_purple_filtered[ii])) / nmp.max(activity_purple_filtered[ii] - nmp.min(activity_purple_filtered[ii]))

range_purple_filtered = range(len(activity_purple_filtered))
plt.subplot(131)
plt.title(str(specimen) + '_purple', size = 40)
for ii in purple_unknown:
    plt.plot(activity_purple_filtered_norm[ii]*10+16*range_purple_filtered[ii], 'C0')
    plt.text(-375, range_purple_filtered[ii]*16, str(ii), size=25)

#### orange
activity_orange_filtered_norm = nmp.zeros_like(activity_orange_filtered)
for ii in range(len(activity_orange_filtered)):
    activity_orange_filtered_norm[ii] = (activity_orange_filtered[ii] - nmp.min(activity_orange_filtered[ii])) / nmp.max(activity_orange_filtered[ii] - nmp.min(activity_orange_filtered[ii]))

range_orange_filtered = range(len(activity_orange_filtered))
plt.subplot(132)
plt.title(str(specimen) + '_orange', size = 40)
for ii in orange_unknown:
    plt.plot(activity_orange_filtered_norm[ii]*10+16*range_orange_filtered[ii], 'C0')
    plt.text(-375, range_orange_filtered[ii]*16, str(ii), size=25)

#### grey
activity_grey_filtered_norm = nmp.zeros_like(activity_grey_filtered)
for ii in range(len(activity_grey_filtered)):
    activity_grey_filtered_norm[ii] = (activity_grey_filtered[ii] - nmp.min(activity_grey_filtered[ii])) / nmp.max(activity_grey_filtered[ii] - nmp.min(activity_grey_filtered[ii]))

range_grey_filtered = range(len(activity_grey_filtered))
plt.subplot(132)
plt.title(str(specimen) + '_grey', size = 40)
for ii in grey_unknown:
    plt.plot(activity_grey_filtered_norm[ii]*10+16*range_grey_filtered[ii], 'C0')
    plt.text(-375, range_grey_filtered[ii]*16, str(ii), size=25)

#### green
# activity_green_filtered_norm = nmp.zeros_like(activity_green_filtered)
# for ii in range(len(activity_green_filtered)):
#     activity_green_filtered_norm[ii] = (activity_green_filtered[ii] - nmp.min(activity_green_filtered[ii])) / nmp.max(activity_green_filtered[ii] - nmp.min(activity_green_filtered[ii]))

# range_green_filtered = range(len(activity_green_filtered))
# plt.subplot(133)
# plt.title(str(specimen) + '_green', size = 40)
# for ii in green_unknown:
#     plt.plot(activity_green_filtered_norm[ii]*10+16*range_green_filtered[ii], 'C0')
#     plt.text(-375, range_green_filtered[ii]*16, str(ii), size=25)


# %%% compare known and unknown
# a reiteration of the above section but where both known and unknown fibres
# are plotted, in case some of the original reference MUs did not include
# all MFs actually belonging to the MU
plt.figure(figsize = (60,40))

#### purple
plt.subplot(131)
plt.title(str(specimen) + '_purple', size = 40)
for ii in range(len(activity_purple_filtered)):
    plt.plot(activity_purple_filtered_norm[ii]*10+16*range_purple_filtered[ii], 'C0')
    # plt.text(-250, range_filtered[ii]*16, str(filtered_index[ii]), fontsize=15)
    plt.text(-375, range_purple_filtered[ii]*16, str(ii), size=25)
for ii in purple_unknown:
    plt.plot(activity_purple_filtered_norm[ii]*10+16*range_purple_filtered[ii], 'C1')
    plt.text(-375, range_purple_filtered[ii]*16, str(ii), size=25)

#### orange
plt.subplot(132)
plt.title(str(specimen) + '_orange', size = 40)
for ii in range(len(activity_orange_filtered)):
    plt.plot(activity_orange_filtered_norm[ii]*10+16*range_orange_filtered[ii], 'C0')
    # plt.text(-250, range_filtered[ii]*16, str(filtered_index[ii]), fontsize=15)
    plt.text(-375, range_orange_filtered[ii]*16, str(ii), size=25)
for ii in orange_unknown:
    plt.plot(activity_orange_filtered_norm[ii]*10+16*range_orange_filtered[ii], 'C1')
    plt.text(-375, range_orange_filtered[ii]*16, str(ii), size=25)

#### grey
plt.subplot(132)
plt.title(str(specimen) + '_grey', size = 40)
for ii in range(len(activity_grey_filtered)):
    plt.plot(activity_grey_filtered_norm[ii]*10+16*range_grey_filtered[ii], 'C0')
    # plt.text(-250, range_filtered[ii]*16, str(filtered_index[ii]), fontsize=15)
    plt.text(-375, range_grey_filtered[ii]*16, str(ii), size=25)
for ii in grey_unknown:
    plt.plot(activity_grey_filtered_norm[ii]*10+16*range_grey_filtered[ii], 'C1')
    plt.text(-375, range_grey_filtered[ii]*16, str(ii), size=25)

#### green
# plt.subplot(133)
# plt.title(str(specimen) + '_green', size = 40)
# for ii in range(len(activity_green_filtered)):
#     plt.plot(activity_green_filtered_norm[ii]*10+16*range_green_filtered[ii], 'C0')
#     # plt.text(-250, range_filtered[ii]*16, str(filtered_index[ii]), fontsize=15)
#     plt.text(-375, range_green_filtered[ii]*16, str(ii), size=25)
# for ii in green_unknown:
#     plt.plot(activity_green_filtered_norm[ii]*10+16*range_green_filtered[ii], 'C1')
#     plt.text(-375, range_green_filtered[ii]*16, str(ii), size=25)
