# Authors: Nikhil Garg, Jean Rouat (advisor)
# Original Date(main_patterns.py): April 18th, 2019,Simon Brodeur
# EMG version creation data : August 2nd 2020
# Organization: 3IT & NECOTIS,
# Université de Sherbrooke, Canada

import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from brian2 import pF, nS, mV, ms, nA, linspace
from brian2 import prefs
from brian2.units import ms, um, meter, Hz
from brian2.synapses.synapses import Synapses
from brian2.core.clocks import defaultclock
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.core.network import Network
from brian2.units.allunits import second
from brian2.monitors.statemonitor import StateMonitor
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup
from brian2.groups import NeuronGroup
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from brian2.__init__ import clear_cache
import gc
import matplotlib.pyplot as plt
import time
from microcircuit import Microcircuit
from utilis import *
from args_emg import args as my_args
import pandas as pd
from itertools import product
# import nni
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

# set_device('cpp_standalone')

# TODO: Design a class for reservoir and evaluate_reservoir be the method of such class
def evaluate_reservoir(args):
    nbInputs = 16

    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    pwd = os.getcwd()

    file_directory = pwd + '/dataset/'
    file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
        args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) +str(args.fold)+ ".npz"

    data = np.load(file_directory + file_name, allow_pickle=True)
    spike_times_test_up = data['spike_times_test_up']
    spike_times_test_dn = data['spike_times_test_dn']
    Y_EMG_Test = data['Y_EMG_Test']
    spike_times_train_up = data['spike_times_train_up']
    spike_times_train_dn = data['spike_times_train_dn']
    Y_EMG_Train = data['Y_EMG_Train']

    algorithm = args.learning_algorithm
    topology = args.topology
    online = args.online_flag
    memoryless = args.memoryless_flag
    p_max = args.connection_density
    adaptiveProb = args.adaptiveProb
    noise = args.noise
    excitatoryProb = args.excitatoryProb
    stdp_tau = args.stdp_tau
    stdp_apre = args.stdp_apre

    nbtimepoints = int(args.duration / args.tstep)
    nbneurons = np.prod(args.minicolumnShape) * np.prod(args.macrocolumnShape)

    spike_rate_array_all_input_train = np.ones((nbInputs, nbtimepoints)) * -1  # Dummy spike counts. Would be discarded in last lines
    spike_rate_array_all_input_test = np.ones((nbInputs, nbtimepoints)) * -1

    spike_rate_array_all_train = np.ones( (nbneurons, nbtimepoints)) * -1  # Dummy spike counts. Would be discarded in last lines
    spike_rate_array_all_test = np.ones((nbneurons, nbtimepoints)) * -1

    if (topology == 'random'):
        p_max = p_max * 0.2

    if(topology == 'custom'):
        input_connection_matrix =  np.loadtxt(args.path_input_connections)
        reservoir_connection_matrix = np.loadtxt(args.path_res_connections)

    if (algorithm == 'none'):
        withSTDP = False
        critical = False
    elif (algorithm == 'critical'):
        withSTDP = False
        critical = True
    elif (algorithm == 'stdp'):
        withSTDP = True
        critical = False
    elif (algorithm == 'critical-stdp'):
        withSTDP = True
        critical = True
    else:
        withSTDP = False
        critical = False


    m = Microcircuit(connectivity=topology, macrocolumnShape=args.macrocolumnShape,
                     minicolumnShape=args.minicolumnShape,
                     p_max=p_max, srate=noise * Hz, excitatoryProb=excitatoryProb, delay=' 0*ms',
                     withSTDP=withSTDP, adaptiveProbab=adaptiveProb, stdp_tau=10, stdp_apre=1e-4, wmax=args.wmax,
                     wmin=args.wmin, winitmin=args.winitmin, winitmax=args.winitmax, refractory=args.refractory)

    # Configure CRITICAL learning rule
    targetCbf = args.cbf
    m.S.c_out_ref = targetCbf  # target critical branching factor
    m.S.alpha = args.lr_critical  # learning rate
    m.S.wmax = args.wmax
    m.S.freeze_time_ms = args.freeze_time_ms

    # tau = args.init_tau * ms
    # tau_dev = args.init_tau_dev
    thr = args.init_thr
    thr_dev = args.init_thr_dev

    # m.G.tau = 'tau*(1-tau_dev) + (2*tau_dev*tau*rand())'
    m.G.vt0 = 'thr*(1-thr_dev) + (2*thr_dev*thr*rand())'
    m.G.vt = 'thr*(1-thr_dev) + (2*thr_dev*thr*rand())'
    m.G.v0 = '0'
    # Define the inputs to the microcircuit

    P = SpikeGeneratorGroup(nbInputs, [], [] * ms)

    Si = Synapses(P, m.G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
	                                                c_in_tot_post += w * int(not_refractory_post)''')


    # sources, targets = input_connection_matrix.nonzero()
    # Si.connect(i=sources, j=targets)
    # Si.connect(condition='i==mmidx_post')
    Si.connect(p=args.input_connection_density*len(m.S)/(nbInputs*len(m.G)))

    Si.w = args.win

    logger.info('Number of neurons in the population: %d' % (len(m.G)))
    logger.info('Number of synapses in the population: %d' % (len(m.S)))
    logger.info('Number of input synapses in the population: %d' % (len(Si)))


    prefs.codegen.target = args.target  # Use numpy instead of cython backend for prototyping, you can remove later (slower overall but no compiling)

    if (critical):
        m.S.plastic = True
    else:
        m.S.plastic = False

    spike_times_up = spike_times_train_up
    spike_times_dn = spike_times_train_dn
    labels = Y_EMG_Train

    label_list = []

    # Configure the monitors and initial weights
    # Initial weights for each permutation, the same is used for both train and test
    initial_input_weights = Si.w[:].copy()  # We make a copy of the initial weights so we can reset to them
    initial_reservoir_weights = m.S.w[:].copy()
    initial_threshold = m.G.vt[:].copy()
    initial_cbf = m.G.cbf[:].copy()
    # NOTE: setting a high time resolution increase the stability of the learning rule
    M = SpikeMonitor(m.G, record=True)
    Mi = SpikeMonitor(P, record=True)
    Mg = StateMonitor(m.G, variables=['cbf'], record=True)
    Mt = StateMonitor(m.G, variables=['vt'], record=True)
    Ms = StateMonitor(m.S, 'w', record=True)

    defaultclock.dt = args.dt * ms

    net = Network(m.G, m.S, P, Si, M, Mt, Mi, Mg, Ms)
    net.store(filename='init')

    for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):

        net.restore(filename='init')
        # Weights and threshold gets saved in training set
        sample_start_time = net.t
        m.G.vt = initial_threshold.copy()
        m.G.cbf = initial_cbf.copy()
        Si.w = initial_input_weights.copy()
        if (iteration == 0):
            logger.info('Reset reservoir weights')
            m.S.w = initial_reservoir_weights.copy()
        else:
            if (memoryless == False):
                m.S.w = trained_reservoir_weights.copy()
                logger.info('Loading reservoir weights from previous iteration')
            else:
                m.S.w = initial_reservoir_weights.copy()
                logger.info('Reset reservoir weights')

        times, indices = convert_data_add_format(sample_time_up, sample_time_down)
        P.set_spikes(indices,times*ms)  # Set the spike to the generator with the current simulation time

        # Choose the duration of the training
        duration = args.duration * ms

        logger.info('Simulating for iteration %i' % (iteration + 1))
        net.run(duration, report='text')

        # if (iteration == 1):
           
        #     plt.subplot(311)
        #     plt.plot(m.G.vt, '.k')
        #     plt.ylabel('Vt')
        #     plt.xlabel('Neuron index')
        #     plt.subplot(312)
        #     plt.hist(m.G.vt, 40)
        #     plt.xlabel('Vt')
        #     plt.subplot(313)

        #     meanVt = np.mean(Mt.vt.T, axis=-1)
        #     stdVt = np.std(Mt.vt.T, axis=-1)
        #     plt.plot(Mt.t/ms, meanVt, color='#1B2ACC')
        #     plt.fill_between(Mt.t /ms, meanVt - stdVt, meanVt + stdVt,
        #                      alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Vt')
        #     plt.tight_layout()
        #     plot_name = pwd + '/plots/' + str(args.experiment_name)
        #     plt.savefig(plot_name + 'threshold' + '.png')
        #     plt.clf()

        #     plt.subplot(311)
        #     plt.plot(m.S.w, '.k')
        #     plt.ylabel('Weight')
        #     plt.xlabel('Synapse index')

        #     plt.subplot(312)
        #     plt.hist(m.S.w, 40)
        #     plt.xlabel('Weight')

        #     plt.subplot(313)
        #     meanW = np.mean(Ms.w.T, axis=-1)
        #     stdW = np.std(Ms.w.T, axis=-1)
        #     plt.fill_between(Ms.t /ms, meanW - stdW, meanW + stdW,
        #                      alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)

        #     plt.plot(Ms.t/ms, meanW)
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Weight')
        #     plt.tight_layout()
        #     plt.savefig(plot_name + 'weights' + '.png')
        #     plt.clf()

        #     fig = plt.figure(facecolor='white', figsize=(6, 5))
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.set_xlabel('Time [sec]')
        #     ax.set_ylabel('Average output contributions')
        #     meanCbf = np.mean(Mg.cbf.T, axis=-1)
        #     stdCbf = np.std(Mg.cbf.T, axis=-1)
        #     ax.plot(Mg.t / ms, meanCbf, color='#1B2ACC')
        #     ax.fill_between(Mg.t / ms, meanCbf - stdCbf, meanCbf + stdCbf,
        #                      alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        #     fig.tight_layout()
        #     fig_name = plot_name + 'cfb' + '.png'
        #     fig.savefig(fig_name)

        #     fig = plt.figure(facecolor='white', figsize=(6, 5))
        #     plt.subplot(211)
        #     plt.title('Spiking activity (input)')
        #     plt.plot(Mi.t / ms, Mi.i, '.', color='k', ms=0.5)
        #     plt.ylabel('Neurons')
        #     plt.xlabel('Time [ms]')

        #     plt.subplot(212)
        #     plt.title('Spiking activity (output)')
        #     plt.plot(M.t / ms, M.i, '.', color='k', ms=0.5)
        #     plt.ylabel('Neurons')
        #     plt.xlabel('Time [ms]')
        #     fig.tight_layout()
        #     fig_name = plot_name + 'activity' + '.svg'
        #     fig.savefig(fig_name)

        #     plt.close()

        # Compute population average firing rate
        avgInputFiringRate = len(Mi.i) / (nbInputs * duration)
        avgOutputFiringRate = len(M.i) / (len(m.G) * duration)

        logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
        logger.info('Average reservoir firing rate: %4.2f Hz' % (avgOutputFiringRate))

        rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(Mi.i),
                                                         time_array=np.array((Mi.t / ms) - sample_start_time / ms),
                                                         duration=duration / ms, tstep=200, nbneurons=nbInputs)
        rate_array = recorded_output_to_spike_rate_array(index_array=np.array(M.i),
                                                         time_array=np.array((M.t / ms) - sample_start_time / ms),
                                                         duration=duration / ms, tstep=200, nbneurons=len(m.G))

        spike_rate_array_all_train = np.dstack((spike_rate_array_all_train, rate_array))
        spike_rate_array_all_input_train=np.dstack((spike_rate_array_all_input_train,rate_array_input))
        label_list.append(np.array(labels[iteration]))
        
        # Reset the neuron membrane potential for both training and testing
        m.G.v = 0
        trained_reservoir_weights = m.S.w[:].copy()  # We make a copy of the trained weights
        # Stop the simulation
        net.stop()
        gc.collect()

    spike_rate_array_all_train = spike_rate_array_all_train[:, :, 1:]
    spike_rate_array_all_input_train = spike_rate_array_all_input_train[:,:,1:]
    X_train, Y_train = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_train, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)

    X_input_train, Y_input_train = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_train, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
    print("Number of Train samples : ")
    print(len(X_train))
    


    

    clf_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    clf_lda.fit(X_train, Y_train)
    clf_lda_input = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    clf_lda_input.fit(X_input_train, Y_input_train)

    clf_svm_linear = make_pipeline(StandardScaler(), LinearSVC())
    clf_svm_linear.fit(X_train, Y_train)
    clf_svm_linear_input = make_pipeline(StandardScaler(), LinearSVC())
    clf_svm_linear_input.fit(X_input_train, Y_input_train)


    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf.fit(X_train, Y_train)
    clf_input = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf_input.fit(X_input_train, Y_input_train)

    

    # Testing
    spike_times_up = spike_times_test_up
    spike_times_dn = spike_times_test_dn
    labels = Y_EMG_Test
    label_list = []

    for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):

        net.restore(filename='init')
        sample_start_time = net.t
        # Weights and threshold gets saved in training set
        m.G.vt = initial_threshold.copy()
        m.G.cbf = initial_cbf.copy()
        Si.w = initial_input_weights.copy()

        if (memoryless == False):
            m.S.w = trained_reservoir_weights.copy()
            logger.info('Loading reservoir weights from previous iteration')
        else:
            m.S.w = initial_reservoir_weights.copy()
            logger.info('Reset reservoir weights')

        times, indices = convert_data_add_format(sample_time_up, sample_time_down)
        P.set_spikes(indices, times*ms)  # Set the spike to the generator with the current simulation time

        # Choose the duration of the training
        duration = args.duration * ms

        logger.info('Simulating for iteration %i' % (iteration + 1))
        net.run(duration, report='text')
        # Compute population average firing rate
        avgInputFiringRate = len(Mi.i) / (nbInputs * duration)
        avgOutputFiringRate = len(M.i) / (len(m.G) * duration)

        logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
        logger.info('Average reservoir firing rate: %4.2f Hz' % (avgOutputFiringRate))

        rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(Mi.i),
                                                               time_array=np.array(
                                                                   (Mi.t / ms) - sample_start_time / ms),
                                                               duration=duration / ms, tstep=200, nbneurons=nbInputs)

        rate_array = recorded_output_to_spike_rate_array(index_array=np.array(M.i),
                                                         time_array=np.array((M.t / ms) - sample_start_time / ms),
                                                         duration=duration / ms, tstep=200, nbneurons=len(m.G))
        spike_rate_array_all_test = np.dstack((spike_rate_array_all_test, rate_array))
        spike_rate_array_all_input_test = np.dstack((spike_rate_array_all_input_test, rate_array_input))

        label_list.append(np.array(labels[iteration]))
        # Reset the neuron membrane potential for both training and testing
        m.G.v = 0
        # Stop the simulation
        net.stop()
        gc.collect()

    pwd = os.getcwd()

    spike_rate_array_all_test = spike_rate_array_all_test[:, :, 1:]
    spike_rate_array_all_input_test=spike_rate_array_all_input_test[:,:,1:]
    X_input_test, Y_input_test = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_test, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
    X_test, Y_test = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_test, label_array=label_list,
                                                    tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
    path = pwd + '/dataset/'
    file_name_test = 'recording-' + str(len(m.G)) + '-Adaptation-' + str(adaptiveProb) + '-Exc_ratio-' + str(
        excitatoryProb) + '-' + str(len(m.S)) + '-' + str(algorithm) + '-' + str(topology) + '-noise-' + str(
        noise) + '-memoryless-' + str(memoryless) + '-' + str(p_max) + '-test' + '.npz'
    np.savez(path + file_name_test, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

    lda_score = clf_lda.score(X_test, Y_test)
    print("lda test accuraccy")
    print(lda_score)
    lda_score_input = clf_lda_input.score(X_input_test, Y_input_test)
    print("lda Input test accuraccy")
    print(lda_score_input)

    svm_linear_score = clf_svm_linear.score(X_test, Y_test)
    print("svm linear test accuraccy")
    print(svm_linear_score)
    svm_linear_score_input = clf_svm_linear_input.score(X_input_test, Y_input_test)
    print("svm linear Input test accuraccy")
    print(svm_linear_score_input)


    svm_score = clf.score(X_test, Y_test)
    print("svm radial test accuraccy")
    print(svm_score)
    svm_score_input = clf_input.score(X_input_test, Y_input_test)
    print("svm radial Input test accuraccy")
    print(svm_score_input)

    #Confusion matrix
    # plt.rcParams.update({'font.size': 16})

    predictions = clf.predict(X_test)
    # ax = skplt.metrics.plot_confusion_matrix(Y_test, predictions, normalize=True)
    # plt.savefig('reservoir'+'confusion'+'.svg')
    # plt.clf()
    # #ROC curve
    # plt.rcParams.update({'font.size': 14})
    # predicted_probas = clf.predict_proba(X_test)
    # ax2 = skplt.metrics.plot_roc(Y_test, predicted_probas)
    # plt.savefig('reservoir'+'roc'+'.svg')
    # plt.clf()

    # plt.rcParams.update({'font.size': 16})
    # predictions = clf_input.predict(X_input_test)
    # ax = skplt.metrics.plot_confusion_matrix(Y_input_test, predictions, normalize=True)
    # plt.savefig('encoder_baseline'+'confusion'+'.svg')
    # plt.clf()
    # plt.rcParams.update({'font.size': 14})
    # #ROC curve
    # predicted_probas = clf_input.predict_proba(X_input_test)
    # ax2 = skplt.metrics.plot_roc(Y_input_test, predicted_probas)
    # plt.savefig('encoder_baseline'+'roc'+'.svg')
    # plt.clf()


    nbsynapses = len(m.S)
    firing_rate = avgOutputFiringRate / Hz

    # nni.report_final_result(svm_score)

    return lda_score,lda_score_input,svm_linear_score,svm_linear_score_input,svm_score,svm_score_input, firing_rate, nbsynapses, nbneurons


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()
    print(args.__dict__)


    # params = nni.get_next_parameter()

    # args.init_tau_min = params['tau_min']
    # args.init_tau_max = args.init_tau_min
    # args.init_thr_min = params['thr_min']
    # args.init_thr_max = args.init_thr_min
    # args.excitatoryProb = params['excitatoryProb']
    # args.adaptiveProb = params['adaptiveProb']


    logging.basicConfig(level=logging.DEBUG)

    # Fix the seed of all random number generator
    seed = 50
    random.seed(seed)
    np.random.seed(seed)

    lda_score,lda_score_input,svm_linear_score,svm_linear_score_input,svm_score,svm_score_input, firing_rate, nbsynapses, nbneurons = evaluate_reservoir(args)


    print('Accuraccy output: ' + str(svm_score))
    print('Accuraccy input: ' + str(svm_score_input))


    logger.info('All done.')
