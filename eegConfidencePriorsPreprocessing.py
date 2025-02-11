"""
Created on Thu Nov 21 09:04:14 2019

@authors: kobe desender

EEG Prior Beliefs

Pre-processing
"""

# Import the necessary Python modules:
import os
import numpy as np
import matplotlib.pyplot as plt
import mne   #pip install mne if module isn't found
import getpass
import pandas as pd

# set user-specific working directory
if getpass.getuser() == 'u0136938':  # kobe
    working_dir = '...' #but main wd here
    data_dir = working_dir + 'rawEEG/'

# Which subject do you want to analyze?
sub = '1'

# 1. Loading Data #
# load the channel file
montage = mne.channels.read_custom_montage(working_dir + '/chanlocs_besa.txt')

#we load in the metadata
meta = pd.read_csv(working_dir + '/EEG_Experiment_finalversion/eeg_test/eeg_test_sub%s.csv' %(sub))

#sub 1 switched answer option is the color task, so we switch 0 to 1 and 0 to 1
if sub == '1':
    meta.loc[meta['task'] == 'color', 'cor'] = meta.loc[meta['task'] == 'color', 'cor'].apply(lambda x: x*0 if x == 1 else x+1)

#same for sub 17 but in the orientation task
if sub == '17':
    meta.loc[meta['task'] == 'orientation', 'cor'] = meta.loc[meta['task'] == 'orientation', 'cor'].apply(lambda x: x*0 if x == 1 else x+1)

# Loads your data data file (change to appropriate file name)
raw = mne.io.read_raw_bdf(data_dir + 'eeg/EEG_Andi_sub%s.bdf' %(sub), preload=True)


#change the sample rate, original was 2048
print('Original sampling rate:', raw.info['sfreq'], 'Hz')
raw = raw.resample(1000)
print('New sampling rate:', raw.info['sfreq'], 'Hz')

#channels drop
raw.drop_channels(['EXG1','EXG2', 'EXG3','EXG4','EXG5', 'EXG6', 'EXG7','EXG8'])
raw.set_montage(montage)
print(raw.info) #a full report
print(raw.info['ch_names'])

# 1.B Filtering out line noise
raw.plot_psd(fmax = 300)
raw.notch_filter(np.arange(50,300, 50), filter_length='auto',phase='zero') # van 48-52, met stappen van 48
#inspect the changes of the filter
raw.plot_psd(fmax = 300)

# 2. Creating Epochs #
# store events from the RAW dataset
events = mne.find_events(raw)

meta_copy = meta.copy()
check = events[0:2892,2]
lenghtcheck = len(check)

#now we make an array where we take the index of the trigger for the stimuli + 1, which gives the trigger of the response
answer = []
for i in range(lenghtcheck):
    if check[i] == 71:
        answer.append(check[i+1])

#here we replace strings c and N with numbers in the meta, so that we can compare them with the triggers in the events
meta_copy['resp'] = meta_copy['resp'].replace(["['c']"], 74)
meta_copy['resp'] = meta_copy['resp'].replace(["['n']"], 78)
response = meta_copy['resp']

lenghtanswer = len(answer)
compare_answers = []
#here we loop over the lenght of answers (1080 if everything went right)
for i in range(lenghtanswer):
    #compare trigger with number in metadata, if everything lines up every value should be 0
   compare_answers.append(response[i] - answer[i])
#here we will look whether confidence labels in EEG events match with excel data
conf = []
for i in range(lenghtcheck):
    if check[i] == 71:
        conf.append(check[i+2])

#replace values of confidence to the same value as trigger
meta_copy['cj'] = meta_copy['cj'].replace([1],77)
meta_copy['cj'] = meta_copy['cj'].replace([2],67)
meta_copy['cj'] = meta_copy['cj'].replace([3],69)
meta_copy['cj'] = meta_copy['cj'].replace([4],73)
meta_copy['cj'] = meta_copy['cj'].replace([5],76)
meta_copy['cj'] = meta_copy['cj'].replace([6],75)

confidence = meta_copy['cj']

#values of -170/-171 you can ignore, these are training trials, all other values should be 0
#all values between -7 and 7 that aren't 0 indicate that something is wrong
compare_conf = []
for i in range(lenghtanswer):
   compare_conf.append(confidence[i] - conf[i])

#if they don't line up, we need to check where it goes wrong. For example in participant 11, the first value is already wrong, meaning that it's wrong from the start.
#the easiest way is to run the lines till line 136, there it will say how many events there are, in this case the metadata is 1080 rows, but events only 1073, meaning that the first 7 events didn't save
#so the solution is to delete the first 7 lines in the meta, which can be done by the following line:

#afterwards you can run the check again, now everything should be 0 and 1, we do need to reset the index so that it can line up
if sub == '11':
    meta = meta.drop(meta.index[range(7)])
    meta.reset_index(drop = True,inplace=True)

#the same thing happened for participant 8, where the first 35 trials weren't saved
if sub == '8':
    meta = meta.drop(meta.index[range(36)])
    meta.reset_index(drop = True,inplace=True)

# store events from the RAW dataset
#no baseline here because better for ICA, baselines happens after ICA
epochs = mne.Epochs(raw, events, event_id = {'stimuli':71}, tmin=-0.5, tmax=2.750,
                     proj=False, baseline= None,
                     preload=True, reject=None)

del raw

#only epochs from main trials are used
epochs = mne.Epochs.__getitem__(epochs, meta['running'] == 'main')
main_meta = meta.loc[(meta[('running')] == 'main')]
epochs.metadata = main_meta


# 3. Rerefence to average (which is at electrode POz right now) #
epochs.set_eeg_reference().apply_proj().average()

# plot the ERPs for uncleaned data
#make a temporary b
temp = epochs.copy()
temp.apply_baseline(baseline=(-.5,-.1))
evoked_clean= temp.average()
evoked_clean.plot()

# 4. Remove and interpolate bad channels#
# Mark bad channels and save in excel which ones you exclude
epochs.plot()
# Check list
print(epochs.info['bads'])

# Interpolate the bad channels
epochs.load_data().interpolate_bads(reset_bads=False)

# 5. Visual artifact rejection
# Mark bad trials
epochs.plot(n_epochs = 8, n_channels = 32)

# - Click on an epoch or channel to reject it from the dataset
# - Use keyboard shortcuts to adapt the window size (click help to see a list)
# - First scroll through the entire data set to get a feel for it
# - Adjust the scale to a comfortable level (HELP to see keyboard controls)
# - Then start at the beginning and click on epochs you think should be rejected
# - Don't reject trials with eye blinks or eye movements; we will get these with ICA

# 6. Save the clean data set
epochs.save(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep1.fif' %(sub,sub), overwrite=True)

# 7. Remove eye-blinks using Independent-Component-Analysis
# Load the saved file (not strictly necesarry)
epochs = mne.read_epochs(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep1.fif' %(sub,sub), preload=True)
epochs.set_montage(montage)

# Now we prepare to run ICA
# We need to know how many ICA components we want
# This is the same amount as the number of UNIQUE channels
# We have to take into account that an interpolated channel does not have any unique
# information, since it is made up of information from the surrounding channels
# We also need to subtract 1 channel, because we are using an average reference
# because this leads to all the channels having in common an amount of
# information equal to 1/number of channels
# So, the correct number of components to get out of the ICA =
# Original number of channels - number of interpolated channels - 1

ncomp =  64 - len(epochs.info['bads']) - 1
print(ncomp) #This should be equal to the original number of channels - number of interpolated channels - 1

# create ICA object with desired parameters
ica = mne.preprocessing.ICA(n_components = ncomp)

# Sometimes it's recommended to do the ICA on (strongly) filtered data
epochs_filt = epochs.copy() #create a copy of the epochs
epochs_filt.load_data().filter(l_freq=2, h_freq=40) #apply a (strong) bandpass filter

# do ICA decomposition
good_EEGchannels = epochs.ch_names[0:64]
good_EEGchannels = [elem for elem in good_EEGchannels if elem not in epochs.info['bads']]
ica.fit(epochs_filt,picks=good_EEGchannels)

# Plot the components and make a screenshot
# Their topography
# clicking on a component will mark it as bad for later rejection
ica.plot_components()

# Plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(epochs, picks=3)


# Their time course:
#save which component in excel
epochs.plot()
ica.plot_sources(epochs_filt)


# Identify the blink component (which characteristics make you think this is the blink component?)
# Decide which component(s) to reject ("project out of the data")
# Click on their name; this will turn it grey
# Remember; the component will not be ONLY due to an artifact
# There will also be some brain activity mixed into it; ICA is not perfect
# This is why you want to be very conservative and not project just any
# component out of your data.

#apply this to the original (unfiltered) epochs

epochs_clean = ica.apply(epochs,exclude=[0,8]) #Apply the weights of the ICA to the epochs
epochs_clean.plot() #check your filtered data to see whether the ICA works

#baseline
epochs_clean.apply_baseline(baseline=(-.5,-.1))

# 8. Save the cleaned ICA data set
epochs_clean.save(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep2.fif' %(sub,sub), overwrite=True)

# 9. Final (second) visual artifact rejection
# Now, the data should be completely fine and ready for analysis, but do a final double check!
# Load the saved file (not strictly necesarry)
epochs_clean = mne.read_epochs(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep2.fif' %(sub,sub), preload=True)

#do a final cleaning
epochs_clean.plot(n_epochs = 10, n_channels = 32)

# 9. Save the final data set (hurray)
epochs_clean.save(data_dir + 'PreprocessedData/sub%s/sub%s_cleanedStep3.fif' %(sub,sub), overwrite=True)
#plot ERP of clean data
evoked_clean = epochs_clean.average()
evoked_clean.plot()
