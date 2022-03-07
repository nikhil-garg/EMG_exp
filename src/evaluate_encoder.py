# Author: Nikhil Garg
# Organization: 3IT & NECOTIS,
# Université de Sherbrooke, Canada

import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import gc
import matplotlib.pyplot as plt
import time
from utilis import *
from args_emg import args as my_args
from encode import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict

def evaluate_encoder(args):

    nbInputs = 16
    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    print(args.__dict__)

    spike_times_train_up, spike_times_train_dn, spike_times_test_up, spike_times_test_dn, X_EMG_Train,X_EMG_Test, Y_EMG_Train,Y_EMG_Test,avg_spike_rate = encode(args)

    nbtimepoints = int(args.duration / args.tstep)

    spike_rate_array_all_input_train = np.ones((nbInputs, nbtimepoints)) * -1  # Dummy spike counts. Would be discarded in last lines
    spike_rate_array_all_input_test = np.ones((nbInputs, nbtimepoints)) * -1

    #Training
    spike_times_up = spike_times_train_up
    spike_times_dn = spike_times_train_dn
    labels = Y_EMG_Train
    label_list = []

    for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
        print(iteration)
        times, indices = convert_data_add_format(sample_time_up, sample_time_down)
        rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                         time_array=np.array(times),
                                                         duration=2000, tstep=200, nbneurons=nbInputs)

        spike_rate_array_all_input_train=np.dstack((spike_rate_array_all_input_train,rate_array_input))
        label_list.append(np.array(labels[iteration]))
        gc.collect()

    spike_rate_array_all_input_train = spike_rate_array_all_input_train[:,:,1:]

    X_input_train, Y_input_train = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_train, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
    print("Number of Train samples : ")
    print(len(X_input_train))

    
    # Testing
    spike_times_up = spike_times_test_up
    spike_times_dn = spike_times_test_dn
    labels = Y_EMG_Test
    label_list = []

    for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
        print(iteration)
        times, indices = convert_data_add_format(sample_time_up, sample_time_down)

        rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                               time_array=np.array(times),
                                                               duration=2000, tstep=200, nbneurons=nbInputs)

        spike_rate_array_all_input_test = np.dstack((spike_rate_array_all_input_test, rate_array_input))
        label_list.append(np.array(labels[iteration]))
        gc.collect()

    spike_rate_array_all_input_test=spike_rate_array_all_input_test[:,:,1:]
    
    X_input_test, Y_input_test = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_test, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
    
    print("Number of Test samples : ")
    print(len(X_input_test))


    X_EMG_Train_segmented, Y_EMG_Train_segmented = segment(X_EMG_Train, Y_EMG_Train,tstep= 40, tstart=0, tstop=400)
    X_EMG_Test_segmented, Y_EMG_Test_segmented = segment(X_EMG_Test, Y_EMG_Test, tstep=40, tstart=0, tstop=400)
    X_train = np.mean(X_EMG_Train_segmented, axis=1)
    X_test = np.mean(X_EMG_Test_segmented, axis=1)
    Y_train = Y_EMG_Train_segmented
    Y_test = Y_EMG_Test_segmented


    '''
    Input model for evaluating spike rates features obtained from temporal difference encoding
    '''

    clf_input = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf_input.fit(X_input_train, Y_input_train)
    svm_score_input = clf_input.score(X_input_test, Y_input_test)
    print("Input test accuraccy")
    print(svm_score_input)


    pwd = os.getcwd()
    plot_dir = pwd + '/plots/'
    plt.rcParams.update({'font.size': 16})
    #Confusion matrix
    predictions = clf_input.predict(X_input_test)
    ax = skplt.metrics.plot_confusion_matrix(Y_input_test, predictions, normalize=True)
    plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'confusion'+'.svg')
    plt.clf()
    #ROC curve
    predicted_probas = clf_input.predict_proba(X_input_test)
    ax2 = skplt.metrics.plot_roc(Y_input_test, predicted_probas)
    plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'roc'+'.svg')
    plt.clf()


    '''
    Baseline model for evaluating time domain averaged features obtained from raw signals
    '''
    clf_baseline = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf_baseline.fit(X_train, Y_train)
    svm_score_baseline = clf_baseline.score(X_test, Y_test)
    print("Baseline accuraccy")
    print(svm_score_baseline)

    #Confusion matrix
    predictions = clf_baseline.predict(X_test)
    ax = skplt.metrics.plot_confusion_matrix(Y_test, predictions, normalize=True)
    plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'confusion'+'.svg')
    plt.clf()
    #ROC curve
    predicted_probas = clf_baseline.predict_proba(X_test)
    ax2 = skplt.metrics.plot_roc(Y_test, predicted_probas)
    plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'roc'+'.svg')
    plt.clf()
    


    plot_dataset(X=X_test,y=Y_test,fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_raw_test',args=args)
    plot_dataset(X=np.array(X_input_test),y=np.array(Y_input_test),fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_decoded_test',args=args)
    np.savez_compressed(
        'spike_data.npz',
        X=np.array(X_input_test),
        Y_EMG_Train=np.array(Y_input_test)
    )


    return svm_score_input,avg_spike_rate, svm_score_baseline

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()
    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    svm_score_input,avg_spike_rate, svm_score_baseline= evaluate_encoder(args)
    print('Average spike rate :')
    print(avg_spike_rate)
    print('Accuraccy input: ' + str(svm_score_input))
    logger.info('All done.')
