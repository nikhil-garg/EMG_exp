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

# from pypet.environment import Environment
# from pypet.brian2.parameter import Brian2Parameter, Brian2MonitorResult
# from pypet.utils.explore import cartesian_product
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

from microcircuit import Microcircuit
from utilis import *
from args_emg import args as my_args
# from ax import optimize
import pandas as pd
from itertools import product
# from brian2 import *
# set_device('cpp_standalone')

#from critical.rankorder import generateRankOrderCodedPatterns, plotPatterns, generateRankOrderCodedData

def evaluate_reservoir(args):
	
	print(args.__dict__)
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	pwd=os.getcwd()

	file_directory = pwd+'/dataset/'
	if(args.dataset=='roshambo'):
		filename = 'EMG_dataset_with_spike_time_roshambo.npz'
	elif(args.dataset=='5_class'):
		filename = 'EMG_dataset_with_spike_time_5_class.npz'

	data=np.load(file_directory+filename,allow_pickle=True )
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
	noise = 0
	excitatoryProb = args.excitatoryProb
	stdp_tau = args.stdp_tau
	stdp_apre = args.stdp_apre

	tstep = 200
	tlast = 2000
	nbtimepoints = int(tlast/tstep)+1
	nbneurons = 256
	
	spike_rate_array_all_train = np.ones((nbneurons,nbtimepoints))*-1 #Dummy spike counts. Would be discarded in last lines
	spike_rate_array_all_test = np.ones((nbneurons,nbtimepoints))*-1

	#TODO: Set it to number of classes
	nboutputneurons = 3
	spike_rate_array_all_train_readout = np.ones((nboutputneurons,nbtimepoints))*-1 #Dummy spike counts. Would be discarded in last lines
	spike_rate_array_all_test_readout = np.ones((nboutputneurons,nbtimepoints))*-1


	if(topology=='random'):
		p_max = p_max * 0.2

	if(algorithm=='none'):
		withSTDP = False
		critical = False
	elif(algorithm=='critical'):
		withSTDP = False
		critical = True
	elif(algorithm=='stdp'):
		withSTDP = True
		critical = False
	else:
		withSTDP = True
		critical= True

	m = Microcircuit(connectivity= topology, macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 2],
	                        p_max=p_max, srate=noise * Hz, excitatoryProb=excitatoryProb, delay=' 0*ms',
	                        withSTDP=withSTDP,adaptiveProbab=adaptiveProb, stdp_tau=10 , stdp_apre=1e-4)

	# Configure CRITICAL learning rule
	targetCbf = 1
	m.S.c_out_ref = targetCbf          # target critical branching factor
	m.S.alpha = 0.1                    # learning rate

	# Define the inputs to the microcircuit
	nbInputs = 16

	P = SpikeGeneratorGroup(nbInputs, [], []*ms)
	num_neurons = len(m.G)
	neuron_ids = np.arange(num_neurons)
	Si = Synapses(P, m.G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
	                                                c_in_tot_post += w * int(not_refractory_post)''')


	Si.connect(condition='i==mmidx_post')

	Si.w = '1.5 + 0.15 * rand()'

	logger.info('Number of neurons in the population: %d' % (len(m.G)))
	logger.info('Number of synapses in the population: %d' % (len(m.S)))

	prefs.codegen.target = 'cython'  # Use numpy instead of cython backend for prototyping, you can remove later (slower overall but no compiling)

	if(critical):
		m.S.plastic = True
	else:
		m.S.plastic = False

	spike_times_up = spike_times_train_up
	spike_times_dn = spike_times_train_dn
	labels = Y_EMG_Train
	label_list = []

	#Output Layer

	soft_reset = False

	eqs = '''
	        dv/dt = -(v-v0)/tau: 1  (unless refractory) # membrane potential
	        dvt/dt = -(vt-vt0)/tau_vt : 1               # adaptive threshold
	        tau : second                          # time constant for membrane potential
	        tau_vt : second                       # time constant for adaptive threshold
	        v0 : 1                                # membrane potential reset value
	        vt0 : 1                               # adaptive threshold reset value
	        vti : 1                               # adaptive threshold increment
	    '''

	if soft_reset:
		reset = '''
		v -= vt     # soft-reset membrane potential
		vt += vti   # increment adaptive threshold
		'''
	else:
		reset = '''
		v = v0      # reset membrane potential
		vt += vti   # increment adaptive threshold
		'''

	    # Spike detection
	threshold = 'v > 1.0'
	
	Gout = NeuronGroup(3, model=eqs, reset=reset, threshold=threshold,
	                    refractory=1*ms, method='exact')


	Gout.v = 0.0
	Gout.tau = 20*ms
	Gout.tau_vt = 50 * ms
	Gout.vt0 = '1.0 + 1.0 * rand()'
	Gout.v0 = 0
	Gout.vt = 1.0
	Gout.vti = 0.1

	taupre = taupost = 5*ms
	wmax = 0.1
	Apre = 0.0002
	Apost = -Apre*taupre/taupost*1.02
	Apost *= wmax
	Apre *= wmax

	So = Synapses(m.G, Gout,
	             
			 '''
			 w : 1
			 dapre/dt = -apre/taupre : 1 (event-driven)
			 dapost/dt = -apost/taupost : 1 (event-driven)
			 ''',
			 on_pre='''
			 v_post += w
			 apre += Apre
			 w = clip(w+apost, 0, wmax)
			 ''',
			 on_post='''
			 apost += Apost
			 w = clip(w+apre, 0, wmax)
			 ''', method='linear')
	
	So.connect(p=1)

	#Teaching Neurons
	nbclasses = 3

	Texc = SpikeGeneratorGroup(nbclasses, [], []*ms)
	Tinh = SpikeGeneratorGroup(nbclasses, [], []*ms)

	Stexc = Synapses(Texc, Gout, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
	                                                ''')
	Stexc.connect(condition='i == j')
	Stexc.w = 1

	Stinh = Synapses(Tinh, Gout, model='w : 1', on_pre='''v_post -= w * int(not_refractory_post)
	                                                ''')
	Stinh.connect(condition='i == j')
	Stinh.w = 0.0

	#Implement lateral inhibition
	# Sinh =Synapses(Gout, Gout,
	# 		'''
	# 		w : 1
	# 		''',
	# 		on_pre='''
	# 		v_post -= 0.5
	# 		''', 
	# 		on_post='''
	# 		v_pre -= 0.5
	# 		''', method='linear')
	# Sinh.connect(condition='i != j')
	# Sinh.w = 1

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
	Mo = SpikeMonitor(Gout, record=True)

	defaultclock.dt = 0.5 * ms

	net = Network(m.G, m.S, P, Si,So,Stexc,Stinh,Gout,M,Mt,Mi,Mg,Ms,Mo, Texc,Tinh)
	net.store(filename='init')

	for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):

		net.restore(filename='init')
		#Weights and threshold gets saved in training set
		sample_start_time = net.t
		m.G.vt = initial_threshold.copy()
		m.G.cbf = initial_cbf.copy()
		Si.w = initial_input_weights.copy()
		if(iteration==0):
			logger.info('Reset reservoir weights')
			m.S.w = initial_reservoir_weights.copy()
			logger.info('Reset readout weights')
			So.w = 0.0001

		else:

			So.w = trained_output_weights.copy()
			logger.info('Loading weights from previous iteration')

			if(memoryless==False):
				m.S.w = trained_reservoir_weights.copy()
				logger.info('Loading reservoir weights from previous iteration')
			else:
				m.S.w = initial_reservoir_weights.copy()
				logger.info('Reset reservoir weights')

		times, indices = convert_data_add_format(sample_time_up, sample_time_down)
		P.set_spikes(indices, times + sample_start_time)  # Set the spike to the generator with the current simulation time


		logger.info('Injecting Teacher signals')
		times, indices = create_teacher_spike_train_exc(labels[iteration],2000,10)
		Texc.set_spikes(indices, times*ms + sample_start_time)  # Set the spike to the generator with the current simulation time
		times, indices = create_teacher_spike_train_inh(labels[iteration],2000,10, 1)
		Tinh.set_spikes(indices, times*ms + sample_start_time)  # Set the spike to the generator with the current simulation time
		# Choose the duration of the training
		duration = 2 * second

		logger.info('Simulating for iteration %i' % (iteration+1))
		net.run(duration, report='text')

		if(iteration==1):
			plt.subplot(311)
			plt.plot(m.G.vt, '.k')
			plt.ylabel('Vt')
			plt.xlabel('Neuron index')
			plt.subplot(312)
			plt.hist(m.G.vt, 20)
			plt.xlabel('Vt')
			plt.subplot(313)
			meanVt=np.mean(Mt.vt.T, axis=-1)
			plt.plot(Mt.t, meanVt)
			plt.xlabel('Time (s)')
			plt.ylabel('Vt')
			plt.tight_layout()
			plot_name = str(args.learning_algorithm) + str(args.adaptiveProb) + str(args.connection_density) + str(
				args.stdp_apre) + str(args.stdp_tau) + 'threshold'+'.eps'
			plt.savefig(plot_name)
			# plt.show()
			plt.clf()
			plt.subplot(311)
			plt.plot(m.S.w , '.k')
			plt.ylabel('Weight / gmax')
			plt.xlabel('Synapse index')
			plt.subplot(312)
			plt.hist(m.S.w , 20)
			plt.xlabel('Weight / gmax')
			plt.subplot(313)
			meanW = np.mean(Ms.w.T, axis=-1)
			plt.plot(Ms.t, meanW)
			plt.xlabel('Time (s)')
			plt.ylabel('Weight / gmax')
			plt.tight_layout()
			plot_name = str(args.learning_algorithm)+str(args.adaptiveProb)+str(args.connection_density)+str(args.stdp_apre)+str(args.stdp_tau)+'.eps'
			plt.savefig(plot_name)
			# plt.show()


		# Compute population average firing rate
		avgInputFiringRate = len(Mi.i) / (nbInputs * duration)
		avgOutputFiringRate = len(M.i) / (len(m.G) * duration)
		avgReadoutFiringRate = len(Mo.i) / (len(Gout) * duration)


		logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
		logger.info('Average reservoir firing rate: %4.2f Hz' % (avgOutputFiringRate))
		logger.info('Average output firing rate: %4.2f Hz' % (avgReadoutFiringRate))


		rate_array = recorded_output_to_spike_rate_array(index_array=np.array(M.i), time_array=np.array((M.t/ms)-sample_start_time/ms), tlast=duration/ms, tstep=200, nbneurons=len(m.G))
		spike_rate_array_all_train = np.dstack((spike_rate_array_all_train, rate_array))
		label_list.append(np.array(labels[iteration]))


		rate_array_readout = recorded_output_to_spike_rate_array(index_array=np.array(Mo.i), time_array=np.array((Mo.t/ms)-sample_start_time/ms), tlast=duration/ms, tstep=200, nbneurons=3)
		spike_rate_array_all_train_readout = np.dstack((spike_rate_array_all_train_readout, rate_array_readout))
		#Reset the neuron membrane potential for both training and testing
		print(rate_array_readout)

		m.G.v = 0
		trained_reservoir_weights = m.S.w[:].copy()  # We make a copy of the trained weights
		# Copy the trained weights and threshold for next sample

		logger.info('Copying trained weights for next iteration')
		trained_output_weights = So.w[:].copy()  # We make a copy of the trained weights

		#Stop the simulation
		net.stop()
		gc.collect()

	# file_name = 'recording-'+str(len(m.G))+'-Adaptation-'+str(adaptiveProbab)+'-Exc_ratio-'+str(excitatoryProb)+'-'+str(len(m.S))+'-'+str(algorithm)+'-'+str(topology)+'-noise-'+str(noise)+'-memoryless-'+str(memoryless)+'-'+ str(p_max)+'-train'+'.npz'
	# pwd = os.getcwd()
	# path = pwd + file_name
	spike_rate_array_all_train = spike_rate_array_all_train[:,:,1:]
	spike_rate_array_all_train_readout=spike_rate_array_all_train_readout[:,:,1:]
	X_train, Y_train = spike_rate_array_to_features(spike_rate_array_all_train, label_list)

	X_train_readout, Y_train_readout = spike_rate_array_to_features(spike_rate_array_all_train_readout, label_list)

	clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
	clf.fit(X_train, Y_train)

    #Testing
	spike_times_up = spike_times_test_up
	spike_times_dn = spike_times_test_dn
	labels = Y_EMG_Test
	neuron_data_list = []
	neuron_time_list = []

	readout_data_list = []
	readout_time_list = []
	label_list = []

	for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):

		net.restore(filename='init')
		sample_start_time = net.t
		#Weights and threshold gets saved in training set
		m.G.vt = initial_threshold.copy()
		m.G.cbf = initial_cbf.copy()
		Si.w = initial_input_weights.copy()

		So.w = trained_output_weights.copy()
		logger.info('Loading output weights from previous iteration')

		if(memoryless==False):
			m.S.w = trained_reservoir_weights.copy()
			logger.info('Loading reservoir weights from previous iteration')
		else:
			m.S.w = initial_reservoir_weights.copy()
			logger.info('Reset reservoir weights')

		times, indices = convert_data_add_format(sample_time_up, sample_time_down)
		P.set_spikes(indices, times + sample_start_time)  # Set the spike to the generator with the current simulation time

		# Choose the duration of the training
		duration = 2 * second

		logger.info('Simulating for iteration %i' % (iteration+1))
		net.run(duration, report='text')
		# Compute population average firing rate
		avgInputFiringRate = len(Mi.i) / (nbInputs * duration)
		avgOutputFiringRate = len(M.i) / (len(m.G) * duration)

		logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
		logger.info('Average reservoir firing rate: %4.2f Hz' % (avgOutputFiringRate))
		# pred_class= np.array([avgReadoutFiringRate_1,avgReadoutFiringRate_2,avgReadoutFiringRate_3]).argmax()
		# logger.info('Actual class: %4.0f' % (labels[iteration]+1))
		# logger.info('Predicted class: %4.0f' % (pred_class+1))

		rate_array = recorded_output_to_spike_rate_array(index_array=np.array(M.i), time_array=np.array((M.t/ms)-sample_start_time/ms), tlast=duration/ms, tstep=200, nbneurons=len(m.G))
		spike_rate_array_all_test = np.dstack((spike_rate_array_all_test, rate_array))
		
		rate_array_readout = recorded_output_to_spike_rate_array(index_array=np.array(Mo.i), time_array=np.array((Mo.t/ms)-sample_start_time/ms), tlast=duration/ms, tstep=200, nbneurons=nboutputneurons)
		print(rate_array_readout)
		spike_rate_array_all_test_readout = np.dstack((spike_rate_array_all_test_readout, rate_array_readout))

		label_list.append(np.array(labels[iteration]))
		#Reset the neuron membrane potential for both training and testing
		m.G.v = 0
		#Stop the simulation
		net.stop()
		gc.collect()

	pwd = os.getcwd()

	spike_rate_array_all_test = spike_rate_array_all_test[:,:,1:]
	X_test, Y_test = spike_rate_array_to_features(spike_rate_array_all_test, label_list)
	
	spike_rate_array_all_test_readout = spike_rate_array_all_test_readout[:,:,1:]
	X_test_readout, Y_test_readout = spike_rate_array_to_features(spike_rate_array_all_test_readout, label_list)

	path = pwd + '/dataset/'
	file_name_test = 'recording-'+str(len(m.G))+'-Adaptation-'+str(adaptiveProb)+'-Exc_ratio-'+str(excitatoryProb)+'-'+str(len(m.S))+'-'+str(algorithm)+'-'+str(topology)+'-noise-'+str(noise)+'-memoryless-'+str(memoryless)+'-'+ str(p_max)+'-test'+'.npz'
	np.savez(path+file_name_test,X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,X_train_readout=X_train_readout, X_test_readout=X_test_readout, Y_train_readout=Y_train_readout, Y_test_readout=Y_test_readout )

	svm_score = clf.score(X_test, Y_test)
	print("test accuraccy")
	print(svm_score)
	nbsynapses = len(m.S)
	firing_rate = avgOutputFiringRate

	return svm_score, firing_rate, nbsynapses

if __name__ == '__main__':

	logger = logging.getLogger(__name__)

	args = my_args()
	print(args.__dict__)

	pwd=os.getcwd()
	file_directory = pwd+'/dataset/'
	if(args.dataset=='roshambo'):
		filename = 'EMG_dataset_with_spike_time_roshambo.npz'
	elif(args.dataset=='5_class'):
		filename = 'EMG_dataset_with_spike_time_5_class.npz'

	data=np.load(file_directory+filename,allow_pickle=True )

	logging.basicConfig(level=logging.DEBUG)

	# Fix the seed of all random number generator
	seed = 500
	random.seed(seed)
	np.random.seed(seed)

	df = pd.DataFrame({"learning_algorithm":[],
						"stdp_apre" : [],
						"stdp_tau" : [],
                         "connection_density":[],
                         "nbsynapses":[],
                         "adaptiveProb":[],
                         "firing_rate":[],
                         "svm_score":[],
                         })
	svm_score, firing_rate, nbsynapses = evaluate_reservoir(args)
	df = df.append({ "learning_algorithm":learning_algorithm,
					 "stdp_apre":stdp_apre,
					 "stdp_tau": stdp_tau,
	                 "connection_density":connection_density,
	                 "nbsynapses":nbsynapses,
	                 "adaptiveProb":adaptiveProb,
	                 "firing_rate":firing_rate,
	                 "svm_score":svm_score,
	                 },ignore_index=True)
	df.to_csv('accuracy_single.csv', index=False)
	print('Accuraccy: ' + str(svm_score))


	logger.info('All done.')
