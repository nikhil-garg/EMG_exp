#!/usr/bin/env python
# coding: utf-8

# Script to load data from EMG and generate spike time arrays. The conversion is carried by delta/slope
# spike conversion algorithm.
# Spike time array contains values of spike times in ms.
# Saved arrays :

# X_EMG_Train: Array of the EMG Digital time series data with length = 300
# Y_EMG_Train: Array of the labels of the training data with length = 300
# X_EMG_Test: Array of the EMG Digital time series data with length = 150
# Y_EMG_Test: Array of the labels of the training data with length = 150
# spike_times_train_up: Spike time arrays with upward polarity in ms for X_EMG_Train. length = 300
# spike_times_train_dn: Spike time arrays with downward polarity in ms for X_EMG_Train. length = 300
# spike_times_test_up: Spike time arrays with upward polarity in ms for X_EMG_Test. length = 150
# spike_times_test_dn: Spike time arrays with downward polarity in ms for X_EMG_Test. length = 150

# Author : Nikhil Garg, 3IT Sherbrooke ; nikhilgarg.bits@gmail.com
# Created : 15 July 2020
# Last edited : 12th September 2020

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import butter, lfilter, welch, square  # for signal filtering
from utilis import *
from args_emg import args as my_args

def encode(args):
    # general stuff
    # sampling frequency of MYO
    fs = 200
    VERBOSE = True
    # pwd = os. getcwd()

    if args.dataset == "roshambo":
        data_dir = "dataset/Roshambo/"
        classes = ["rock", "paper", "scissor"]
        classes_dict = {"rock": 0, "paper": 1, "scissor": 2}
        classes_inv = {v: k for k, v in classes_dict.items()}
    elif args.dataset == "5_class":
        data_dir = "dataset/Gesture_5_class/"
        classes = ["pinky", "elle", "yo", "index", "thumb"]
        classes_dict = {"pinky": 0, "elle": 1, "yo": 2, "index": 3, "thumb": 4}
        classes_inv = {v: k for k, v in classes_dict.items()}
    elif args.dataset == "pinch":
        data_dir = "dataset/Pinch/"
        classes = ["Pinch1", "Pinch2", "Pinch3", "Pinch4"]
        classes_dict = {"Pinch1": 0, "Pinch2": 1, "Pinch3": 2, "Pinch4": 3}
        classes_inv = {v: k for k, v in classes_dict.items()}
    else:
        print("Invalid dataset")

    args.classes_dict = classes

    class Person(object):
        def __init__(self, name, emg, ann, classes=classes):
            self.name = name
            self.emg = emg
            self.ann = ann
            self.trials = {c: [] for c in classes}
            self.begs = {c: [] for c in classes}
            self.ends = {c: [] for c in classes}
            self.x = {c: [] for c in classes}
            self.y = {c: [] for c in classes}
            self.ts = {c: [] for c in classes}
            self.pol = {c: [] for c in classes}
            self.emg_spks = {c: [] for c in classes}
            self.spk_trials = {c: [] for c in classes}

    subjects = {}
    names = sorted([name for name in os.listdir(data_dir) if "emg" in name])

    for name in names:
        _emg = np.load(data_dir + "{}".format(name)).astype("float32")
        _ann = np.concatenate(
            [
                np.array(["none"]),
                np.load(data_dir + "{}".format(name.replace("emg", "ann")))[:-1],
            ]
        )
        subjects["_".join(name.split("_")[:2])] = Person(
            name.split("_")[0], _emg, _ann, classes=classes
        )
        print(
            "Loaded {}: EMG = [{}] // ANN = [{}]".format(
                "_".join(name.split("_")[:2]), _emg.shape, len(_ann)
            )
        )
    for name, data in subjects.items():
        for _class in classes:
            _annotation = np.float32(data.ann == _class)
            derivative = np.diff(_annotation) / 1.0
            begins = np.where(derivative == 1)[0]
            ends = np.where(derivative == -1)[0]
            for b, e in zip(begins, ends):
                _trials = data.emg[b:e]
                data.trials[_class].append(_trials / np.std(_trials))
                data.begs[_class].append(b)
                data.ends[_class].append(e)
    print("Done sorting trials!")

    # check that we get 5 trials per subject per gesture
    for sub_name, sub_data in subjects.items():
        for _class, trials in sub_data.trials.items():
            assert len(trials) == 5, "Something wrong with the number of trials!"
    print("All good!")

    X_EMG = []
    Y_EMG = []
    SUB_EMG = []
    SES_EMG = []
    TRI_EMG = []

    for name, data in subjects.items():
        for gesture in classes:
            for trial in range(5):
                X_EMG.append(data.trials[gesture][trial])
                Y_EMG.append(classes_dict[gesture])
                SUB_EMG.append(int(name[7:9]))
                SES_EMG.append(int(name[17:19]))
                TRI_EMG.append(trial)
    X_EMG = np.array(X_EMG)
    Y_EMG = np.array(Y_EMG)
    SUB_EMG = np.array(SUB_EMG)
    SES_EMG = np.array(SES_EMG)
    TRI_EMG = np.array(TRI_EMG)

    # X_EMG_uniform is a time series data array with length of 400. The initial segments are about 397, 493 etc which
    # makes it incompatible in some cases where uniform input is desired.

    nb_trials = X_EMG.shape[0]
    len_trial = fs * 2  # 2 seconds of trial, sampling rate is 200
    nb_channels = 8
    X_EMG_uniform = np.ones((nb_trials, len_trial, nb_channels))
    for i in range(len(X_EMG)):
        trial_length = X_EMG[i].shape[0]
        if trial_length > len_trial:
            X_EMG_uniform[i] = X_EMG[i][0:len_trial]
        elif trial_length < len_trial:
            short = len_trial - trial_length
            pad = np.zeros((short, nb_channels))
            X_EMG_uniform[i] = np.append(X_EMG[i], pad, axis=0)
        else:
            X_EMG_uniform[i] = X_EMG[i]
    # print(len(X_EMG))
    print("Number of samples in dataset:")
    print(len(X_EMG_uniform))
    print(len(Y_EMG))
    print("Class labels:")
    print(list(set(Y_EMG)))
    print("Subjects : ")
    print(list(set(SUB_EMG)))
    print("Sessions : ")
    print(list(set(SES_EMG)))
    print("Trials per session : ")
    print(list(set(TRI_EMG)))

    # Take session 0,1 as train and session 2 as test.

    X_EMG_Train = []
    Y_EMG_Train = []
    X_EMG_Test = []
    Y_EMG_Test = []
    for i in range(len(Y_EMG)):
        if (SES_EMG[i]) == args.fold:
            X_EMG_Test.append(X_EMG_uniform[i])
            Y_EMG_Test.append(Y_EMG[i])
        else:
            X_EMG_Train.append(X_EMG_uniform[i])
            Y_EMG_Train.append(Y_EMG[i])
    interpfact = args.encode_interpfact
    refractory_period = args.encode_refractory  # in ms
    th_up = args.encode_thr_up
    th_dn = args.encode_thr_dn
    n_ch = 8
    fs = 200

    # Generate the training data
    spike_times_train_up = []
    spike_times_train_dn = []
    for i in range(len(X_EMG_Train)):
        spk_up, spk_dn = gen_spike_time(
            time_series_data=X_EMG_Train[i],
            interpfact=interpfact,
            fs=fs,
            th_up=th_up,
            th_dn=th_dn,
            refractory_period=refractory_period,
        )
        spike_times_train_up.append(spk_up)
        spike_times_train_dn.append(spk_dn)
    # Generate the test data
    spike_times_test_up = []
    spike_times_test_dn = []
    for i in range(len(X_EMG_Test)):
        spk_up, spk_dn = gen_spike_time(
            time_series_data=X_EMG_Test[i],
            interpfact=interpfact,
            fs=fs,
            th_up=th_up,
            th_dn=th_dn,
            refractory_period=refractory_period,
        )
        spike_times_test_up.append(spk_up)
        spike_times_test_dn.append(spk_dn)

    rate_up_test = gen_spike_rate(spike_times_test_up)
    rate_dn_test = gen_spike_rate(spike_times_test_up)
    rate_up_train = gen_spike_rate(spike_times_train_up)
    rate_dn_train = gen_spike_rate(spike_times_train_up)
    avg_spike_rate = (rate_up_test+rate_dn_test+rate_up_train+rate_dn_train)/4
    print("Average spiking rate")
    print(avg_spike_rate)


    _t = np.arange(
        0, 2000, 5
    )  # Time array of 2000ms for the 200 samples per second. For ploting purpose.
    _t_spike = np.arange(0, 2000, 1)  # Time array for defining the X axis of graph.

    # Plot a up segment
    plt.eventplot(spike_times_test_up[1], color=[0, 0, 1], linewidth=0.5)
    plt.xlabel("Time(ms)")
    plt.ylabel("Channel")
    plt.title("Spike raster plot for up channel")

    # Plot a dn segment
    plt.eventplot(spike_times_test_dn[1], color=[1, 0, 0], linewidth=0.5)
    plt.xlabel("Time(ms)")
    plt.ylabel("Channel")
    plt.title("Spike raster plot for down channel")

    channels = np.linspace(0, 7, num=8)
    plt.plot(_t, X_EMG_Test[1], linewidth=0.5)
    plt.legend(channels)
    plt.title("Raw Data")
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude")
    pwd = os.getcwd()
    fig_dir = pwd + '/plots/'
    fig_name = 'encoded-data' + str(args.dataset) + str(args.encode_thr_up) + str(args.encode_thr_dn) + str(
        args.encode_refractory) + str(args.encode_interpfact)+str(args.fold) + ".svg"

    plt.savefig(fig_dir+fig_name)
    plt.clf()


    spike_times_test_up = np.array(spike_times_test_up)
    spike_times_test_up = np.array(spike_times_test_up)
    spike_times_train_up = np.array(spike_times_train_up)
    spike_times_train_up = np.array(spike_times_train_up)

    file_path = "dataset/"
    file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
        args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact)+ str(args.fold) + ".npz"
    # np.savez(file_path, X_EMG_Train=X_EMG_Train, Y_EMG_Train=Y_EMG_Train,X_EMG_Test=X_EMG_Test,Y_EMG_Test=Y_EMG_Test,spike_times_train_up = spike_times_train_up ,spike_times_train_dn = spike_times_train_dn,spike_times_test_up = spike_times_test_up ,spike_times_test_dn = spike_times_test_dn)
    # np.savez('/home/turing/Desktop/EMG/EMG_dataset_with_spike_time.npz', X_EMG_Train=X_EMG_Train, Y_EMG_Train=Y_EMG_Train,X_EMG_Test=X_EMG_Test,Y_EMG_Test=Y_EMG_Test,spike_times_train_up = spike_times_train_up ,spike_times_train_dn = spike_times_train_dn,spike_times_test_up = spike_times_test_up ,spike_times_test_dn = spike_times_test_dn)
    np.savez_compressed(
        file_path + file_name,
        X_EMG_Train=X_EMG_Train,
        Y_EMG_Train=Y_EMG_Train,
        X_EMG_Test=X_EMG_Test,
        Y_EMG_Test=Y_EMG_Test,
        spike_times_train_up=spike_times_train_up,
        spike_times_train_dn=spike_times_train_dn,
        spike_times_test_up=spike_times_test_up,
        spike_times_test_dn=spike_times_test_dn,
    )
    return spike_times_train_up, spike_times_train_dn, spike_times_test_up, spike_times_test_dn,X_EMG_Train,X_EMG_Test, Y_EMG_Train,Y_EMG_Test,avg_spike_rate

if __name__ == '__main__':
    args = my_args()
    print(args.__dict__)
    # Fix the seed of all random number generator
    encode(args)
