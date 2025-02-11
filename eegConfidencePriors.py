# -*- coding: utf-8 -*-
"""
Analysis of the preprocessed data of the EEG version of Van Marcke et al. 2024
Fully within-subjects: all participants have + and - feedback, and easy vs hard training difficulty.
Data collected by Andi Smet

@author: Kobe Desender
"""

###########################
### 1. SETTING UP STUFF ###
###########################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import getpass
import pandas as pd
import scipy.stats
import pickle
import math


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    LinearModel,
    get_coef,
)

if getpass.getuser() == 'u0136938':  # kobe
    working_dir = '...' #home folder of the project
    data_dir = working_dir + 'Experiment/eeg data/'
    fig_dir = working_dir + 'eeg figures/'


def decoding_cluster_test(X):
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    
    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=-1, verbose=False)
    
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0])
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_)

def tfr_cluster_test(X):
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    
    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=-1, verbose=False)
    
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0])
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)

#Default style
plt.style.use('fast') #https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

# Set global font size
plt.rcParams.update({
    'font.size': 16,             # Global font size
    'axes.titlesize': 16,        # Title font size
    'axes.labelsize': 16,        # Label font size
    'xtick.labelsize': 16,       # X-axis tick font size
    'ytick.labelsize': 16,       # Y-axis tick font size
    'legend.fontsize': 16,       # Legend font size
    'figure.titlesize': 18       # Overall figure title font size
})


sys.path.append(working_dir+"eeg scripts")
from spearman_func import scorer_spearman, _parallel_scorer, _check_y, repeated_spearman, repeated_corr #custom functions, borrowed from JR, for spearman decoding

# load the channel file
montage = mne.channels.read_custom_montage(working_dir + '/chanlocs_besa.txt')

#Some useful stuff for throughout the code
subs = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','29','31','32']
picks_f = ["Fz", "F1", "F2"] #fronto-central ROI (Lim et al.)
picks_p = ["CPz","Pz","POz","P1","P2"] #centro-parietal ROI


#Whether to analyze stimulus-locked or response-locked data
locking = "stim" # "stim" or "resp" or "cj"

###########################
### 2. LOAD DATA        ###
###########################
# Load all data and put into one big dict called epochs
# OR reload the previously saved epochs
if (os.path.exists(data_dir+'epochs_stimLocked.pkl')) & (locking=="stim"): #load stimulus-locked data from file
    times = np.linspace(-.2, .8, 11)
    with open(data_dir +'epochs_stimLocked.pkl', 'rb') as file:
        epochs = pickle.load(file)
elif (os.path.exists(data_dir+'epochs_respLocked.pkl')) & (locking=="resp"): #load response-locked data from file
    times = np.linspace(-.2, .7, 10)
    with open(data_dir +'epochs_respLocked.pkl', 'rb') as file:
        epochs = pickle.load(file)
elif (os.path.exists(data_dir+'epochs_cjLocked.pkl')) & (locking=="cj"): #load response-locked data from file
    times = np.linspace(-.7, .1, 9)
    with open(data_dir +'epochs_cjLocked.pkl', 'rb') as file:
        epochs = pickle.load(file)
else:
    #Create stimulus-locked epochs
    epochs_stim = {}
    for idx, sub in enumerate(subs):
        tempDat = mne.read_epochs(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep3.fif'  %(sub,sub), preload=True).set_montage(montage)

        #Exclude non-relevant electrodes
        tempDat.drop_channels(["Status","Iz"])

        #Downsample to 500hz to save memory
        tempDat.resample(sfreq=500)

        epochs_stim[idx] = tempDat

    #Response-locked data
    if locking=="resp":

        epochs_resp = {}

        # Loop through each trial and create response-locked epochs
        for idx, sub in enumerate(subs):
            print('response locking data of subject %s' %(sub))
            tempDat = []
            for i, rt in enumerate(epochs_stim[idx].metadata['rt']):
                if rt < (2 - 1/epochs_stim[idx].info['sfreq']):
                    # Shift the time axis of the existing stimulus-locked epoch
                    # Note 1: RTs are more precise then sfreq so round to same precision
                    # Note 2: epochs runs until 2.75; we subtract rt + .75s, so max(RT) <  (2.75 - .75) = 2

                    temp = epochs_stim[idx][i].copy().shift_time(-round(rt * epochs_stim[idx].info['sfreq']) / epochs_stim[idx].info['sfreq'],relative=True)

                    # Crop the epoch to -.2 until .75
                    temp.crop(tmin=-.2, tmax=.75)

                    # The baseline gives issues, even thought it's already baselined
                    temp.baseline = None

                    # Append the response-locked epoch to the list
                    tempDat.append(temp)

            # Concatenate the list of response-locked epochs into a single object
            epochs_resp[idx] = mne.concatenate_epochs(tempDat)

    #CJ-locked data
    if locking=="cj":

        epochs_cj = {}

        # Loop through each trial and create CJ-locked epochs
        # Note that rtCONF is relative to rt and not to stimulus
        for idx, sub in enumerate(subs):
            print('cj locking data of subject %s' %(sub))
            tempDat = []
            rtcj_full = epochs_stim[idx].metadata['rt'] + epochs_stim[idx].metadata['RTconf']/1000
            for i, rtcj in enumerate(rtcj_full):
                if rtcj < (2.65 - 1/epochs_stim[idx].info['sfreq']):
                    # Note 1: RTs are more precise then sfreq so round to same precision
                    # Note 2: epochs runs until 2.75; we subtract rtcj_full + .1s, so max(rtcj_full) <  (2.75 - .1) = 2.65
                    temp = epochs_stim[idx][i].copy().shift_time(-round(rtcj * epochs_stim[idx].info['sfreq']) / epochs_stim[idx].info['sfreq'],relative=True)

                    # Crop the epoch to -.75 until .1
                    temp.crop(tmin=-.750, tmax=.1)

                    # The baseline gives issues, even thought it's already baselined
                    temp.baseline = None

                    # Append the response-locked epoch to the list
                    tempDat.append(temp)

            # Concatenate the list of response-locked epochs into a single object
            epochs_cj[idx] = mne.concatenate_epochs(tempDat)

#Save epochs_stim for easier use later on (only done the 1st time so commented out here)
#with open(data_dir+'epochs_stimLocked%d.pkl' %csd_trans, 'wb') as file:
#    pickle.dump(epochs_stim, file)
#Save epochs_resp for easier use later on
#with open(data_dir+'epochs_respLocked%d_work.pkl' %csd_trans, 'wb') as file:
#    pickle.dump(epochs_resp, file)
#Save epochs_cj for easier use later on
#with open(data_dir+'epochs_cjLocked%d_work.pkl' %csd_trans, 'wb') as file:
#    pickle.dump(epochs_cj, file)

#Participants 4 only pressed "cj=2" in the last 2 blocks, so exclude these blocks (i.e. the orientation task)
epochs[3] = epochs[3][epochs[3].metadata['task']!="orientation"]

#Combine cj 1 and 2
for idx, sub in enumerate(subs): 
    #combine 1 and 2
    epochs[idx].metadata['cj'][epochs[idx].metadata['cj']<2] = 2
    #reschule from 1-5
    epochs[idx].metadata['cj'] = epochs[idx].metadata['cj']-1
    
# Define A Priori time windows
if locking=="stim":
    p3_time = np.arange(np.where(epochs[0].times==0.43999999999999995)[0][0],np.where(epochs[0].times==0.6399999999999999)[0][0]) #200ms window around .54 (i.e. the peak)
if locking=="resp":
    pe_time = np.arange(np.where(epochs[0].times==.3)[0][0],np.where(epochs[0].times==.5)[0][0]) #☻200ms time window 300-500 post-response
if locking=="cj":
    fr_time = np.arange(np.where(epochs[0].times==-.5)[0][0],np.where(epochs[0].times==-.3)[0][0]) #

###########################
### 3. ERPS             ###
###########################

#cut-off epochs to be in line with the decoding length
for idx, sub in enumerate(subs):
    if locking=="stim":
        epochs[idx].crop(tmin=-.1,tmax=.7)
    if locking=="resp":
        epochs[idx].crop(tmin=-.1,tmax=.7)
    if locking=="cj":
        epochs[idx].crop(tmin=-.7,tmax=.1)

# Plots for confidence
picks=picks_p #or picks_f for the frontal signal
X = np.empty([len(subs),len(epochs[0].times),2])
for idx, sub in enumerate(subs):
   X[idx,:,0] = epochs[idx][epochs[idx].metadata['cj']<4].pick(picks).get_data().mean(axis=(0,1))
   X[idx,:,1] = epochs[idx][epochs[idx].metadata['cj']>3].pick(picks).get_data().mean(axis=(0,1))
plt.figure(figsize=(5, 5))
plt.xlim(epochs[0].times[0], epochs[0].times[-1])
plt.ylim(-3*1e-6, 5.5*1e-6)
plt.axhline(y=0, color='grey', linestyle='dashed');plt.axvline(x=0, color='grey', linestyle='dashed');plt.ylabel('µV');plt.xlabel('Time (s)')
plt.plot(epochs[0].times, X[:,:,0].mean(axis=0),color="#7CAE00",label='Low confidence')
line = plt.fill_between(epochs[0].times,X[:,:,0].mean(axis=0)+(X[:,:,0].std(axis=0)/np.sqrt(len(subs))),X[:,:,0].mean(axis=0)-(X[:,:,0].std(axis=0)/np.sqrt(len(subs))),alpha=.2,color="#7CAE00")
plt.plot(epochs[0].times, X[:,:,1].mean(axis=0),color="#C77CFF",label='High confidence')
plt.fill_between(epochs[1].times,X[:,:,1].mean(axis=0)+(X[:,:,1].std(axis=0)/np.sqrt(len(subs))),X[:,:,1].mean(axis=0)-(X[:,:,1].std(axis=0)/np.sqrt(len(subs))),alpha=.2,color="#C77CFF")

#do and add the stats
t, c, ps, H = mne.stats.permutation_cluster_1samp_test(X[:,:,0]-X[:,:,1])
for idx, p in enumerate(ps):
    if p<.05:
        plt.axhline(y=-.0000025,xmin=c[idx][0][0]/len(epochs[0].times),xmax=c[idx][0][-1]/len(epochs[0].times),color='black',linewidth=2.5) #this function takes input between 0 and 1, so scale by len(times)
        print('Significant cluster from %.2f ms until %.2f, p<%.4f' % (epochs[0].times[c[idx][0][0]], epochs[0].times[c[idx][0][-1]],p))
 

# Plots for prior beliefs
picks=picks_f #or picks_p for frontal signal
X = np.empty([len(subs),len(epochs[0].times),2])
for idx, sub in enumerate(subs):
   X[idx,:,0] = epochs[idx][(epochs[idx].metadata['fb_name']=="easy") | (epochs[idx].metadata['fb_name']=="positivefb")].pick(picks).get_data().mean(axis=(0,1))
   X[idx,:,1] = epochs[idx][(epochs[idx].metadata['fb_name']=="hard") | (epochs[idx].metadata['fb_name']=="negativefb")].pick(picks).get_data().mean(axis=(0,1))
plt.figure(figsize=(5, 5))
plt.xlim(epochs[0].times[0], epochs[0].times[-1])
plt.ylim(-2*1e-6, 5*1e-6)
plt.axhline(y=0, color='grey', linestyle='dashed');plt.axvline(x=0, color='grey', linestyle='dashed');plt.legend();plt.ylabel('µV');plt.xlabel('Time (s)')
plt.plot(epochs[0].times, X[:,:,0].mean(axis=0),color='#F8766D',label='positive prior')
plt.fill_between(epochs[0].times,X[:,:,0].mean(axis=0)+(X[:,:,0].std(axis=0)/np.sqrt(len(subs))),X[:,:,0].mean(axis=0)-(X[:,:,0].std(axis=0)/np.sqrt(len(subs))),alpha=.2,color='#F8766D')
plt.plot(epochs[0].times, X[:,:,1].mean(axis=0),color='#00BFC4',label='negative prior')
plt.fill_between(epochs[1].times,X[:,:,1].mean(axis=0)+(X[:,:,1].std(axis=0)/np.sqrt(len(subs))),X[:,:,1].mean(axis=0)-(X[:,:,1].std(axis=0)/np.sqrt(len(subs))),alpha=.2,color='#00BFC4')
#do and add the stats
t, c, ps, H = mne.stats.permutation_cluster_1samp_test(X[:,:,0]-X[:,:,1])
for idx, p in enumerate(ps):
    if p<.05:
        plt.axhline(y=-.0000015,xmin=c[idx][0][0]/len(epochs[0].times),xmax=c[idx][0][-1]/len(epochs[0].times),color='black',linewidth=2.5)
        print('Significant cluster from %.2f ms until %.2f' % (epochs[0].times[c[idx][0][0]], epochs[0].times[c[idx][0][-1]]))



#################################
### 4. SINGLE-TRIAL ESTIMATES ###
#################################
for idx, sub in enumerate(subs):
    print('computing data of subject %s' %(sub))
    
    # Interpolate bad electrodes 
    epochs[idx].info['bads'] = ['P6']
    if locking=="resp":
            epochs[idx].info['bads'] = ['P8','P6']
    epochs[idx].interpolate_bads()

    X = epochs[idx].get_data() #EEG signals: n_epochs, n_channels, n_times
    y_cat = epochs[idx].metadata['cj']>3 #confidence categorical
    y_cont = np.array(epochs[idx].metadata['cj']) #confidence, continouus
    acc = np.array(epochs[idx].metadata['cor']) #accuracy
    diff = np.array(epochs[idx].metadata['trialdifflevel']) # difficulty level, if needed
    
    rt = np.array(epochs[idx].metadata['rt']) #rt
    rtcj = np.array(epochs[idx].metadata['RTconf']) #rtcj

    epochs[idx].metadata['condition'] = 'positive'
    epochs[idx].metadata['condition'][epochs[idx].metadata['fb_name'] == 'negativefb']  = 'negative'
    epochs[idx].metadata['condition'][epochs[idx].metadata['fb_name'] == 'hard']  = 'negative'
    
    #1.4.6. Average amplitude in the P3/PE/frontal time window at posterior/frontal electrodes
    if locking=="stim":
        X_P3 = epochs[idx].get_data(picks=picks_p) #P3
        Average_P3_amplitude = X_P3[:,:,p3_time].mean(axis=(1,2)) #p3_time, a 200ms window around the peak p3
        X_P3 = epochs[idx].get_data(picks=picks_f) #frontal P3
        frontal_P3_amplitude = X_P3[:,:,p3_time].mean(axis=(1,2)) #p3_time, a 200ms window around the peak p3
    if locking=="resp":
        X_PE = epochs[idx].get_data(picks=picks_p) #Pe
        Average_PE_amplitude_300500 = X_PE[:,:,pe_time].mean(axis=(1,2)) #pe time, 300-500
    if locking=="cj":
        X_fr = epochs[idx].get_data(picks=picks_f) #frontal signal
        Average_fr_amplitude = X_fr[:,:,fr_time].mean(axis=(1,2)) #-500:-300
    
    #SAVE ALL OF THIS INTO ONE BIG DF to perform-single trial analysis on (in R, because python doesn't do mixed models :/
    df = pd.DataFrame()
    df.insert(0,"sub",np.repeat(sub,len(y_cont)))
    df.insert(1,"cj",y_cont)
    df.insert(2,"rt",rt)
    df.insert(3,"cor",acc)
    df.insert(4,"rtcj",rtcj)
    if locking=="stim":
        df.insert(5,"average_P3_amplitude",Average_P3_amplitude) #Option 2: avearge amplitude (450-650)
    if locking=="resp":
        df.insert(5,"average_PE_amplitude",Average_PE_amplitude_300500) #Option 2: avearge amplitude (300-500)
    if locking=="cj":
        df.insert(5,"average_fr_amplitude",Average_fr_amplitude) #Option 2: avearge amplitude (x-x)
    df.insert(6,"difficulty",diff) #trial difficulty
    df.insert(7,"condition",np.array(epochs[idx].metadata['fb_name']) ) #condition
    if locking=="stim":
        df.insert(8,"frontal_P3_amplitude",frontal_P3_amplitude) #
    
    #And save the DF
    if locking=="stim":
        df.to_csv(data_dir +'eeg data for Herregods model/Sub' + str(sub) +'_stim.csv')
    if locking=="resp":
        df.to_csv(data_dir +'eeg data for Herregods model/Sub' + str(sub) +'_resp.csv')
    if locking=="cj":
        df.to_csv(data_dir +'eeg data for Herregods model/Sub' + str(sub) +'_cj.csv')



#################################
### 5. DECODING               ###
#################################
#Note, this has been run on a cluster computer in parallel
for idx, sub in enumerate(subs):
    print('decoding locking data of subject %s' %(sub))
    
    #cut-off epochs (if not done before)
    if locking=="stim":
        epochs[idx].crop(tmin=-.1,tmax=.7)
    if locking=="resp":
        epochs[idx].crop(tmin=-.1,tmax=.7)
    if locking=="cj":
        epochs[idx].crop(tmin=-.7,tmax=.1)

    # Interpolate bad electrode 
    epochs[idx].info['bads'] = ['P6']
    if locking=="resp":
            epochs[idx].info['bads'] = ['P8','P6']
    epochs[idx].interpolate_bads()
    
    #extra relevant information
    X = epochs[idx].get_data() #EEG signals: n_epochs, n_channels, n_times
    y_cont = np.array(epochs[idx].metadata['cj']) #confidence decoding, continouus
    epochs[idx].metadata['condition'] = 'positive'
    epochs[idx].metadata['condition'][epochs[idx].metadata['fb_name'] == 'negativefb']  = 'negative'
    epochs[idx].metadata['condition'][epochs[idx].metadata['fb_name'] == 'hard']  = 'negative'
    y_prior = np.array(epochs[idx].metadata['condition'] == "positive") #convert to array, easier below

    # Sub-select only trials that are matched in confidence, to rule out we're decoding connfidence
    mask = np.repeat(False,len(y_prior))
    for t in range(len(y_prior)):
        if mask[t] == False: #if trial is still unchosen, find a partner trials
            # Find all trials where y_prior is not equal to y_prior[t] and y_cont is equal to y_cont[t]
            indices = np.where((y_prior[t] != y_prior) & (y_cont[t] == y_cont) & (mask == False))[0]
            if len(indices)>0:
                mask[np.random.choice(indices)] = True
                mask[t] = True
    #mask.mean() # % of trials retained
    #pd.crosstab(y_prior[mask],y_cont[mask]) #should be the same confidence trials
        
    # 1.4.1 Categorical decoding of prior beliefs
    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))) #set up a logistic decoded
    time_decod = mne.decoding.GeneralizingEstimator(clf, n_jobs=4, scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X[mask,:,:], y_prior[mask], cv=5, n_jobs=4) #cross-validation
    # Save mean scores across cross-validation splits
    np.save(data_dir +'crossdecoding/priorbeliefs_Sub' + str(sub) +'_%s.npy' %(locking), np.mean(scores, axis=0))
    
    # 1.4.2. Continous decoding for confidence, based on Senoussi's "mpt_eeg_rating_predict.py"
    clf = sklearn.pipeline.Pipeline([('scaler', StandardScaler()),('linReg', LinearModel(Ridge())) ])
    gat = mne.decoding.GeneralizingEstimator(clf, scoring=make_scorer(scorer_spearman),n_jobs=-1)
    cv = StratifiedKFold(5)
    scores = cross_val_multiscore(gat, X[mask,:,:], y_cont[mask], cv = cv, n_jobs = 4,verbose = True)
    # Save mean scores across cross-validation splits
    np.save(data_dir +'crossdecoding/cj_Sub' + str(sub) +'_%s.npy' %(locking), np.mean(scores, axis=0))
   
    # 1.4.3 Cross decoding continuous confidence => prior beliefs (scoring rule= predicted_cj[prior==1]-predicted_cj[prior==0]
    clf = sklearn.pipeline.Pipeline([('scaler', StandardScaler()),('linReg', LinearModel(Ridge())) ])
    gat = mne.decoding.GeneralizingEstimator(clf, scoring=make_scorer(scorer_spearman),n_jobs=-1)
    #implement 5-fold cross-validation manually
    idxs = np.repeat(np.arange(1, 6), repeats= math.ceil(len(y_cont)/5)) #create the folds
    np.random.shuffle(idxs) #shuffle the folds
    idxs = idxs[:len(y_cont)] #match with len(y)
    scores = np.ones((X.shape[2],X.shape[2], 5))
    for f in range(1,6):
        gat.fit(X[(idxs!=f) & (mask==True),:,:], y_cont[(idxs!=f) & (mask==True)]) #train on confidence, then test on prior
        pred_cont = gat.predict(X[(idxs==f) & (mask==True),:,:]) #get the actual continuous prediction; shape=[trials, time1, time2]
        scores[:,:,f-1] = np.mean(pred_cont[y_prior[(idxs==f) & (mask==True)]==True,:,:],axis=0)-np.mean(pred_cont[y_prior[(idxs==f) & (mask==True)]==False,:,:],axis=0) #average predicted continuous on positive - negative prior beliefs (across time)
    # Save mean scores across cross-validation splits
    np.save(working_dir+'/cjTrain_priorTest_Sub' + str(sub) +'_allLockings.npy', np.mean(scores, axis=2))


#1.5.1. All lockings generalizing across each other
tempgen_cj = np.empty(shape=(len(subs),483,483))
tempgen_prior = np.empty(shape=(len(subs),483,483))
tempgen_cross = np.empty(shape=(len(subs),483,483))
for idx, sub in enumerate(subs):
    tempgen_prior[idx,:,:] = np.load(data_dir +'crossdecoding/priorbeliefs_Sub' + str(sub) +'_allLockings.npy')
    tempgen_cj[idx,:,:] = np.load(data_dir +'crossdecoding/cj_Sub' + str(sub) +'_allLockings.npy')
    tempgen_cross[idx,:,:] = np.load(data_dir +'crossdecoding/cjTrain_priorTest_Sub' + str(sub) +'_allLockings.npy')

# Create seperate plots for each type 
t_decoding = np.arange(-.1,.7+1.6,.01) #3 times -.1 until .7

# First, for tempgen_cj
fig,ax = plt.subplots(figsize=(8, 6))
im1 = ax.imshow(tempgen_cj.mean(axis=0), interpolation="lanczos", origin="lower", cmap="RdBu_r", extent=t_decoding[[0, -1, 0, -1]],vmin=-np.abs(tempgen_cj.mean(axis=0)).max(),vmax=np.abs(tempgen_cj.mean(axis=0)).max())
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization (confidence)")
ax.axvline(.7, color="white");ax.axhline(.7, color="white") #transition between epochs
ax.axvline(1.5, color="white");ax.axhline(1.5, color="white")
ax.axvline(0, color="k", linestyle=':');ax.axhline(0, color="k", linestyle=':') #zero points
ax.axvline(x=0.8, color="k", linestyle=':', ymin=1/3, ymax=1);ax.axhline(y=.8, color="k", linestyle=':',xmin=1/3,xmax=1)
ax.axvline(x=2.2, color="k", linestyle=':', ymin=2/3, ymax=1);ax.axhline(y=2.2, color="k", linestyle=':',xmin=2/3,xmax=1)
ax.set_xticks(np.arange(-.1,.7+1.6,.2));ax.set_xticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
ax.set_yticks(np.arange(-.1,.7+1.6,.2));ax.set_yticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
cbar1 = plt.colorbar(im1)
cbar1.set_label("Spearman rho")
# Add contours where p_values are significant
p_values = decoding_cluster_test(np.array(tempgen_cj)).T
np.unique(p_values[p_values<.05])
p_values = p_values<.05
contours = ax.contour(p_values, levels=[0], colors='black', linewidths=2,extent=t_decoding[[0, -1, 0, -1]])

# Second, tempgen_prior
fig,ax = plt.subplots(figsize=(8, 6))
im1 = ax.imshow(tempgen_prior.mean(axis=0), interpolation="lanczos", origin="lower", cmap="RdBu_r", extent=t_decoding[[0, -1, 0, -1]],vmin=.5-np.abs(tempgen_prior.mean(axis=0)-.5).max(),vmax=.5+np.abs(tempgen_prior.mean(axis=0)-.5).max())
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization (prior beliefs)")
ax.axvline(.7, color="white");ax.axhline(.7, color="white") #transition between epochs
ax.axvline(1.5, color="white");ax.axhline(1.5, color="white")
ax.axvline(0, color="k", linestyle=':');ax.axhline(0, color="k", linestyle=':') #zero points
ax.axvline(x=0.8, color="k", linestyle=':', ymin=1/3, ymax=1);ax.axhline(y=.8, color="k", linestyle=':',xmin=1/3,xmax=1)
ax.axvline(x=2.2, color="k", linestyle=':', ymin=2/3, ymax=1);ax.axhline(y=2.2, color="k", linestyle=':',xmin=2/3,xmax=1)
ax.set_xticks(np.arange(-.1,.7+1.6,.2));ax.set_xticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
ax.set_yticks(np.arange(-.1,.7+1.6,.2));ax.set_yticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
cbar1 = plt.colorbar(im1)
cbar1.set_label("AUC")
# Add contours where p_values are significant
p_values = decoding_cluster_test(np.array(tempgen_prior)-.5).T
np.unique(p_values[p_values<.05])
p_values = p_values<.05
contours = ax.contour(p_values, levels=[True,False], colors='black', linewidths=2,extent=t_decoding[[0, -1, 0, -1]])

# Third, tempgen across-decoding 
fig,ax = plt.subplots(figsize=(8, 6))
im1 = ax.imshow(tempgen_cross.mean(axis=0), interpolation="lanczos", origin="lower", cmap="RdBu_r", extent=t_decoding[[0, -1, 0, -1]],vmin=-np.abs(tempgen_cross.mean(axis=0)).max(),vmax=np.abs(tempgen_cross.mean(axis=0)).max())
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Cross-decoding (Cont confidence => prior beliefs)")
ax.axvline(.7, color="white");ax.axhline(.7, color="white") #transition between epochs
ax.axvline(1.5, color="white");ax.axhline(1.5, color="white")
ax.axvline(0, color="k", linestyle=':');ax.axhline(0, color="k", linestyle=':') #zero points
ax.axvline(x=0.8, color="k", linestyle=':', ymin=1/3, ymax=1);ax.axhline(y=.8, color="k", linestyle=':',xmin=1/3,xmax=1)
ax.axvline(x=2.2, color="k", linestyle=':', ymin=2/3, ymax=1);ax.axhline(y=2.2, color="k", linestyle=':',xmin=2/3,xmax=1)
ax.set_xticks(np.arange(-.1,.7+1.6,.2));ax.set_xticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
ax.set_yticks(np.arange(-.1,.7+1.6,.2));ax.set_yticklabels(np.concatenate([np.arange(-0.1, 0.7, 0.2), np.arange(-0.1, 0.7, 0.2), np.arange(-0.7, 0.1, 0.2)]).round(1)) # Labels for the ticks
cbar1 = plt.colorbar(im1)
cbar1.set_label("predicted confidence (positive) - predicted confidence (negative)")
# Add contours where p_values are significant
p_values = decoding_cluster_test(np.array(tempgen_cross)-.5).T
np.unique(p_values[p_values<.05])
p_values = p_values<.05
contours = ax.contour(p_values, levels=[True,False], colors='black', linewidths=2,extent=t_decoding[[0, -1, 0, -1]])



#################################
### 6. TIME-FREQUENCY STUFF   ###
#################################
#cut-off epochs to be in line with the decoding, except that you want the pre-stimulus period
for idx, sub in enumerate(subs):
    if locking=="stim":
        epochs[idx].crop(tmin=-.5,tmax=.7) #note, including the baseline
    if locking=="resp":
        epochs[idx].crop(tmin=-.1,tmax=.7) 
    if locking=="cj":
        epochs[idx].crop(tmin=-.7,tmax=.1) 

#What do you want to analyze?
runWhat = "priors" # "cj" or "priors", which condition do you want?
picks = picks_p #which electrodes do you want, parietal or frontal

frequencies = np.arange(3,31,1)  # Frequencies from 3 to 30 Hz (cause the epoch is too short)        
n_cycles = np.linspace(1.5, 5,num=len(frequencies))  # Different number of cycles per frequency, so 1.5cycle for 2h, 5 cycles for 20hz

# Initialize  a list to store the power differences for each subject
power_diff_list = []

# Loop through each participant's epochs
from scipy.ndimage import gaussian_filter
for idx, sub in enumerate(subs):
    print('TFRing data of subject %s' %(sub))
    if runWhat == "priors":
        epochs_l = epochs[idx][(epochs[idx].metadata['fb_name']=="easy") | (epochs[idx].metadata['fb_name']=="positivefb")]
        epochs_h = epochs[idx][(epochs[idx].metadata['fb_name']=="hard") | (epochs[idx].metadata['fb_name']=="negativefb")]
    else:
        epochs_h = epochs[idx][(epochs[idx].metadata['cj']>4)]
        epochs_l = epochs[idx][(epochs[idx].metadata['cj']<5)]
    
    #select the relevant electrodes    
    epochs_l = epochs_l.pick(picks) # Run the TFR for each of these electrodes and average AFTER otherwise you risk messing up different phases (ie. mehdi)
    epochs_h = epochs_h.pick(picks)
    
    # Compute TFR for each set of epochs
    tfr_l = mne.time_frequency.tfr_morlet(epochs_l, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    tfr_h = mne.time_frequency.tfr_morlet(epochs_h, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

    # Compute the difference in power, per electrode 
    power_diff = ((tfr_h.data - tfr_l.data) / (tfr_l.data+tfr_h.data)) * 100 #percentage change
    # Average over electrodes
    power_diff = np.mean(power_diff,axis=0)
    #smooth the data a bit using a gaussian filter
    power_diff = gaussian_filter(power_diff, 2) 
    
    #Finally, save this in a list of power differences
    power_diff_list.append(power_diff)


# Compute the grand average of a list of power differences
tfr_diff_avg =  np.expand_dims(np.mean(power_diff_list, axis=0),axis=0)
info = mne.create_info(ch_names=['CPz'], sfreq=tfr_l.info['sfreq'], ch_types=['eeg']) #create info object with 1 channel (otherwise it conflicts)
tfr_diff_avg = mne.time_frequency.AverageTFR(info=info, data=tfr_diff_avg , times=tfr_l.times, freqs=tfr_l.freqs, nave=len(power_diff_list))

# Plot the difference
if locking=='stim':
    fig,ax = plt.subplots(figsize=(9, 5)) #because longer time-window, -500:700
else:
    fig,ax = plt.subplots(figsize=(6, 5)) #time window of -100:700
if runWhat == "priors":
    header="Power difference (prior beliefs)"
else:
    header="Power difference (confidence)"
tfr_diff_avg.plot(picks='all', title=header,cmap="RdBu_r",axes=ax,vmin=-11,vmax=11)
plt.axhline(y=0, color='grey', linestyle='dashed');plt.axvline(x=0, color='grey', linestyle='dashed')

#run stats
X = np.stack(power_diff_list)  #make this into a np array
X = np.squeeze(X)

# Add contours where p_values are significant
p_values = tfr_cluster_test(X)
np.unique(p_values[p_values<.05])
p_values.min()
p_values = p_values < .04
extent = [tfr_diff_avg.times.min(), tfr_diff_avg.times.max(), tfr_diff_avg.freqs.min(), tfr_diff_avg.freqs.max()]
plt.contour(p_values, levels=[0], colors='black', linewidths=2,extent=extent,origin='lower')
