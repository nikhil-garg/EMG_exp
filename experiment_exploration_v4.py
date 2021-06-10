
import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from evaluate_reservoir import *
from utilis import *
from args_emg import args as my_args
import pandas as pd
from itertools import product
import time
from encode import *

if __name__ == '__main__':

	args = my_args()
	print(args.__dict__)
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(__name__)

	df = pd.DataFrame({"memoryless":[],
						"win":[],
					   "input_connection_density":[],
						"dataset":[],
						"tstart" : [],
						"tlast":[],
						"freeze_time_ms":[],
						"refractory":[],
						"learning_algorithm":[],
						"macrocolumnShape":[],
						"minicolumnShape":[],
						"topology":[],
						"stdp_apre" : [],
						"stdp_tau" : [],
						"wmax": [],
						"wmin":[],
						"winitmax":[],
						"winitmin":[],
						"cbf":[],
						"lr_critical":[],
                        "connection_density":[],
                        "nbneurons":[],
                        "nbsynapses":[],
                        "adaptiveProb":[],
                        "excitatoryProb":[],
                        "noise":[],
                        "firing_rate":[],
                        "lda_score":[],
                        "lda_score_input":[],
                        "svm_linear_score":[],
                        "svm_linear_score_input":[],
                        "svm_rbf_score":[],
					    "svm_rbf_score_input":[],
					    "fold":[]
                         })

	parameters = dict(
		dataset=['5_class']
		,memoryless=[True]
		,win=['1.0 * rand()']
		,input_connection_density=[0.15]
		,tstart = [0]
		,tlast = [1800]
		,freeze_time_ms = [0]
		,thr_init = [1]
		,thr_init_dev = [0.5]
		,refractory=[1]
        ,learning_algorithm=['None']
        ,topology = ['small-world']
        ,lr_critical = [0.1]
        ,macrocolumnShape=[[1,1,1],[1,1,2],[1,2,2],[2,2,2],[2,2,3],[2,2,4],[2,3,4],[2,4,4],[3,4,4],[4,4,4]]
        ,minicolumnShape=[[4,4,2]]
        ,connection_density=[0.1]
        ,adaptiveProb=[1]
        ,excitatoryProb=[0.8]
        ,noise=[0]
        ,stdp_tau = [25]
        ,stdp_apre = [1e-3]
		,wmax = [1]
		,winitmax=[0.25]
		,winitmin=[0]
		,cfb=[1]
		,fold=[1]
    )
	param_values = [v for v in parameters.values()]

	for args.dataset,args.memoryless_flag, args.win,args.input_connection_density,args.tstart,args.tlast,args.freeze_time_ms,args.init_thr, args.init_thr_dev,args.refractory, args.learning_algorithm,args.topology,args.lr_critical, args.macrocolumnShape,args.minicolumnShape, args.connection_density, args.adaptiveProb,args.excitatoryProb,args.noise, args.stdp_tau, args.stdp_apre, args.wmax, args.winitmax, args.winitmin,args.cbf,args.fold in product(*param_values):

			# Fix the seed of all random number generator
		seed = int(args.seed)
		random.seed(seed)
		np.random.seed(seed)
		# args.experiment_name = str(args.path_res_connections)+str(args.path_input_connections)+str(args.memoryless_flag)+str(args.input_connection_density)+str(args.tstart)+str(args.tlast)+str(args.learning_algorithm) + str(args.winitmax)+str(args.winitmin)+str(args.connection_density)+str(args.wmax)+str(args.refractory)+str(args.fold)
		lda_score,lda_score_input,svm_linear_score,svm_linear_score_input,svm_score,svm_score_input, firing_rate, nbsynapses, nbneurons = evaluate_reservoir(args)
		df = df.append({ "dataset":args.dataset,
						 "memoryless":args.memoryless_flag,
						 "win" :args.win,
						 "input_connection_density":args.input_connection_density,
						 "tstart":args.tstart,
						 "tlast" : args.tlast,
						 "freeze_time_ms":args.freeze_time_ms,
						 "thr":args.init_thr,
						 "thr_dev":args.init_thr_dev,
						 "refractory" : args.refractory,
						 "learning_algorithm":args.learning_algorithm,
						 "macrocolumnShape":args.macrocolumnShape,
						 "minicolumnShape":args.minicolumnShape, 
						 "topology":args.topology,
						 "stdp_apre":args.stdp_apre,
						 "stdp_tau": args.stdp_tau,
						 "wmax":args.wmax,
						 "wmin":args.wmin,
						 "winitmax":args.winitmax,
						 "winitmin":args.winitmin,
						 "cbf":args.cbf,
						 "lr_critical":args.lr_critical,
		                 "connection_density":args.connection_density,
 		                 "nbneurons":nbneurons,
		                 "nbsynapses":nbsynapses,
		                 "adaptiveProb":args.adaptiveProb,
		                 "excitatoryProb":args.excitatoryProb,
		                 "noise":args.noise,
		                 "firing_rate":firing_rate,
		                 "lda_score":lda_score,
		                 "lda_score_input":lda_score_input,
		                 "svm_linear_score":svm_linear_score,
		                 "svm_linear_score_input":svm_linear_score_input,
		                 "svm_rbf_score":svm_score,
						 "svm_rbf_score_input":svm_score_input,
						 "fold":args.fold,
						 "seed":args.seed
		                 },ignore_index=True)
		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+'.csv'
		pwd = os.getcwd()

		if args.log_file_path is None:
			log_dir = pwd+'/log_dir/'
			df.to_csv(log_dir+log_file_name, index=False)
		else : 
			log_dir = args.log_file_path
			df.to_csv(log_dir+log_file_name, index=False)

	df.to_csv(log_file_name, index=False)
	logger.info('All done.')
