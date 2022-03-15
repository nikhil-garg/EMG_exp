
import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

from evaluate_reservoir import *
from utilis import *
from args_emg import args as my_args
from evaluate_encoder import  *
from itertools import product
import time

if __name__ == '__main__':

	args = my_args()
	print(args.__dict__)
	# Fix the seed of all random number generator
	seed = 50
	random.seed(seed)
	np.random.seed(seed)
	df = pd.DataFrame({	"dataset":[],
						"encode_thr_up":[],
						"encode_thr_dn":[],
						"encode_refractory" :[],
						"encode_interpfact":[],
                        "firing_rate":[],
                        "svm_score":[],
                        "svm_score_baseline":[]
                         })

	parameters = dict(
		dataset = [ '5_class','roshambo']
		,threshold = [0.5]
		,interpfact = [5]
		,refractory = [1]
		, fold=[1,2,3]
    )
	param_values = [v for v in parameters.values()]

	for args.dataset,threshold,interpfact,refractory,fold in product(*param_values):

		args.encode_thr_up = threshold
		args.encode_thr_dn = threshold
		args.encode_refractory = refractory
		args.encode_interpfact = interpfact
		args.fold=fold
		args.experiment_name = str(args.dataset)+str(threshold)+str(interpfact)+str(refractory)+str(fold)

		svm_score, firing_rate, svm_score_baseline = evaluate_encoder(args)
		df = df.append({ "dataset":args.dataset,
						 "fold":args.fold,
						 "encode_thr_up":args.encode_thr_up,
						 "encode_thr_dn":args.encode_thr_dn,
						 "encode_refractory": args.encode_refractory,
						 "encode_interpfact": args.encode_interpfact,
		                 "firing_rate":firing_rate,
		                 "svm_score":svm_score,
		                 "svm_score_baseline":svm_score_baseline
		                 },ignore_index=True)

		timestr = time.strftime("%Y%m%d-%H%M%S")
		log_file_name = 'accuracy_log'+str(timestr)+'.csv'
		pwd = os.getcwd()
		log_dir = pwd+'/log_dir/'
		df.to_csv(log_dir+log_file_name, index=False)

	df.to_csv(log_file_name, index=False)
	# logger.info('All done.')
