import numpy as np
import scipy as sc
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt


def convert_data_add_format(spike_time_array_1, spike_time_array_2):
    '''
    Function to convert multichannel UP and DN spike time array list(Generated from encoding) to index & time array format(AER). 

    Parameters
    ----------
    spike_time_array_1 : 1st list of spike time arrays. Each element of this list is single channel of data. 
    spike_time_array_2 : 2nd list of spike time arrays

    Returns
    -------
    time_array : 1D array with spike times in ms
    index_array : 1D array with index number of neruon

    '''
    #Obtain the number of channels in the spike time array
    nbchannels = len(spike_time_array_1)
    #Initialize the time and index array
    time_array = []
    index_array = []
    #Loop through all the channels of 1st array
    for i in range(nbchannels):
        #Add the time stamps for channel i. 
        time_array = np.hstack((time_array, spike_time_array_1[i]))
        #Add neuron index for channel i. 
        index_array = np.hstack(
            (index_array, np.ones((len(spike_time_array_1[i]))) * i)
        )
    #Loop through all the channels of 2nd array
    for i in range(nbchannels):
        time_array = np.hstack((time_array, spike_time_array_2[i]))
        #The neuron index of the second array begins from nbchannels
        index_array = np.hstack(
            (index_array, np.ones((len(spike_time_array_2[i]))) * (i + nbchannels))
        )
    return time_array, index_array


def recorded_output_to_spike_rate_array(
        index_array, time_array, duration, tstep, nbneurons
):
    '''
    
    Convert AER spikes to rate array

    Parameters
    ----------
    index_array : 1D array with index of neuron
    time_array : 1D array with spike times
    duration : Trial duration in ms.
    tstep : Time step for which rate has to be calculated. In ms. 
    nbneurons : Total number of neurons

    Returns
    -------
    spike_rate_array : Array of shape (nbneurons, nbtimepoints)

    '''
    
    time_points = np.arange(0, duration, tstep)
    nbtimepoints = len(time_points)
    spike_rate_array = np.ones((nbneurons, nbtimepoints))
    for n in range(nbneurons):
        #Array index for nth neuron
        array_index = np.where(index_array == n)
        for step in range(nbtimepoints+1):
            #step 0 has no spikes. 
            if step < 1:
                spike_rate_array[n, step] = 0
            else:
                #Obtain spike times for nth neuron
                spike_times_neuron_n = time_array[array_index]
                #Obtain the spike times for past one step
                array_indeces_past_one_steps = np.where(
                    (spike_times_neuron_n > tstep * (step - 1))
                    & (spike_times_neuron_n <= tstep * (step))
                )
                #Obtain the number of spikes
                nb_spike_times_past_one_steps = len(array_indeces_past_one_steps[0])
                # Compute rate in spikes per second and assign to spike_rate_array[n, step-1]
                spike_rate_array[n, step-1] = nb_spike_times_past_one_steps * ( 1000 / (tstep) )  
    return spike_rate_array


def spike_rate_array_to_features(spike_rate_array, label_array, tstart, tlast, tstep):
    '''
    Function to generate feature vectors from spike rate arrays. 


    Parameters
    ----------
    spike_rate_array : List of spike rate arrays. Each item corresponds to single sample
    label_array : List of labels. Each item corresponds to label of single sample
    tstart : Time stamp from which spikes has to be considered for features
    tlast : Time stamp for calculation of last feature vector. 
    tstep : Time step for 

    Returns
    -------
    x : List of features
    y : List of labels

    '''
    x = []
    y = []
    nbsamples = spike_rate_array.shape[2] # Total number of samples in spike rate array
    nbtimepoints = spike_rate_array[:, :, 1].shape[1] #Time points in a single sample
    nbneurons = spike_rate_array[:, :, 1].shape[0] #Number of neurons in a single sample

    for i in range(nbsamples):
        for j in range(nbtimepoints):
            # Only accept the window starting from tstart to tstop
            if (j >= tstart/tstep) & (j <= tlast/tstep):
                x.append(spike_rate_array[:, j, i])
                y.append(label_array[i])
    return x, y


def gen_spike_time(time_series_data, interpfact, fs, th_up, th_dn, refractory_period):
    
    '''
    Function to generate spike time array from time series data

    Parameters
    ----------
    time_series_data : Multi channel time series data 
    interpfact : Interpolation factor for spike encoding
    fs : Sampling frequency
    th_up : Threshold UP
    th_dn : Threshold DN
    refractory_period : Refractory period for spike encoding

    Returns
    -------
    spike_time_array_up : Spike time array for UP channel
    spike_time_array_dn : Spike time array for DN channel 

    '''
    spike_time_array_up = []
    spike_time_array_dn = []
    #Iterate through all the channels in the time series data
    for channel_number in range((time_series_data.shape)[1]):
        #Single channel data
        raw_channel = time_series_data[:, channel_number]
        #Array with time stamp for each sample in ms
        _t = 1000 * np.arange(0, raw_channel.shape[0] / fs, 1.0 / fs)
        #Encoded data for single channel
        spk_up, spk_dn = signal_to_spike_refractory(
            interpfact, _t, raw_channel, th_up, th_dn, refractory_period
        )
        spike_time_array_up.append(spk_up)
        spike_time_array_dn.append(spk_dn)
    return spike_time_array_up, spike_time_array_dn

def signal_to_spike_refractory(
        interpfact, time, amplitude, thr_up, thr_dn, refractory_period
):
    '''

    Function definition to convert single channel time series array to spike times.

    Parameters
    ----------
    interpfact : Interpolation factor for spike encoding
    time : Time stamps of the data array. (Amplitudes)
    amplitude : Analog data for encoding
    thr_up : Threshold(UP)
    thr_dn : Threshold(DN)
    refractory_period : Refractory period for spike encoding

    Returns
    -------
    spike_up : Array of spike times in ms. 
    spike_dn : Array of spike times in ms.

    '''
    actual_dc = 0
    spike_up = []
    spike_dn = []
    last_sample = interpfact * refractory_period

    f = sc.interpolate.interp1d(time, amplitude)
    rangeint = np.round((np.max(time) - np.min(time)) * interpfact)

    # Interpolate the time series analog signal to represent 1 second/ 1000ms of data into 'interpolate' samples

    xnew = np.linspace(np.min(time), np.max(time), num=int(rangeint), endpoint=True)
    data = np.reshape([xnew, f(xnew)], (2, len(xnew))).T

    i = 0
    while i < (len(data) - int(last_sample)):
        if (actual_dc + thr_up) < data[i, 1]:
            spike_up.append(data[i, 0])  # spike up
            actual_dc = data[i, 1]  # update current dc value
            i += int(refractory_period * interpfact)
        elif (actual_dc - thr_dn) > data[i, 1]:
            spike_dn.append(data[i, 0])  # spike dn
            actual_dc = data[i, 1]  # update current value
            i += int(refractory_period * interpfact)
        else:
            i += 1
    return spike_up, spike_dn


def gen_spike_rate(spike_time_array):
    '''
    Function to calculate average rate for given spike time array

    Parameters
    ----------
    spike_time_array 

    Returns
    -------
    spike_rate

    '''
    nb_trials = len(spike_time_array)
    nb_electrodes = len(spike_time_array[1])
    spike_sum = 0
    for trial_number in range(nb_trials):
        for channel_number in range(nb_electrodes):
            spike_sum = spike_sum + len(spike_time_array[trial_number][channel_number])
            time_max = 2
    av_spike_sum = spike_sum / (nb_trials * nb_electrodes)
    spike_rate = av_spike_sum / time_max
    return spike_rate


def segment(X, Y, tstep=40, tstart=0, tstop=400):
    '''

    Parameters
    ----------
    X
    Y
    tstep
    tstart
    tstop

    Returns
    -------
    X_semgented
    Y_segmented

    '''
    num_samples = len(X)
    X_segmented = []
    Y_segmented = []
    num_segments = int((tstop-tstart)/tstep)
    for i in range (num_samples):
        X_i = X[i]
        Y_i = Y[i]
        for j in range(num_segments): 
            X_segmented.append(X_i[(tstart + j*tstep):(tstart + (j+1)*tstep),:])
            Y_segmented.append(Y_i)
    return X_segmented, Y_segmented

def plot_dataset(X,y,fig_dir, fig_name,args):
    '''

    Parameters
    ----------
    X
    y
    fig_dir
    fig_name

    Returns
    -------
    ax

    '''
    pca = PCA(random_state=1)
    pca.fit(X)

    nbfeatures = X.shape[1]
    features = np.linspace(1,nbfeatures,num=nbfeatures)
    
    clf = RandomForestClassifier()
    clf.fit(X,y)

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    padding = np.arange(len(features)) + 0.5
    plt.clf()
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature index")
    plt.title("Importance of features(i.e. Channel numbers)")

    plt.savefig(fig_dir+fig_name+'_feature_importances'+'.png')
    plt.clf()

    skplt.decomposition.plot_pca_2d_projection(pca, X, y, biplot=False)
    plt.savefig(fig_dir+fig_name+'_pca'+'.svg')
    plt.clf()

    return sorted_idx


def create_teacher_spike_train_exc(true_label, length_ms, period_ms):
    '''

    Parameters
    ----------
    true_label
    length_ms
    period_ms

    Returns
    -------
    time_array
    index_array

    '''
    nbtimepoints = int(length_ms / period_ms)
    index_array = np.ones(nbtimepoints) * true_label
    time_array = np.arange(0, length_ms, period_ms)

    return time_array, index_array


def create_teacher_spike_train_inh(true_label, length_ms, period_ms, inh_factor):
    '''

    Parameters
    ----------
    true_label
    length_ms
    period_ms
    inh_factor

    Returns
    -------
    time_array
    index_array

    '''
    all_labels = [0, 1, 2]
    iter_flag = 0
    for label in all_labels:
        if label != true_label:
            nbtimepoints_inh = int(length_ms / (period_ms * inh_factor))
            index_array_inh = np.ones(nbtimepoints_inh) * label
            time_array_inh = np.arange(0, length_ms, period_ms * inh_factor)
            if iter_flag != 0:
                index_array = np.hstack((index_array, index_array_inh))
                time_array = np.hstack((time_array, time_array_inh))
            else:
                index_array = index_array_inh
                time_array = time_array_inh
            iter_flag += 1
    return time_array, index_array

