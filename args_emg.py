# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:29:29 2020

@author: Nikhil
"""

import argparse


# data_dir = '../EMG_CRITICAL/Raw_dataset/Gesture_5_class/'
# classes = ['pinky', 'elle', 'yo', 'index', 'thumb']
# classes_dict = {'pinky': 0, 'elle': 1, 'yo': 2, 'index': 3, 'thumb': 4}


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on EMG data"
    )

    # Defining the model
    parser.add_argument(
        "--dataset", default="roshambo", type=str, help="Dataset(roshambo or 5_class)"
    )
    parser.add_argument(
        "--encode_thr_up",
        default=0.5,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_thr_dn",
        default=0.5,
        type=float,
        help="Threshold UP for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_refractory",
        default=1,
        type=float,
        help="Refractory period for spike encoding" "e.g. 0.25, 0.50 etc.",
    )
    parser.add_argument(
        "--encode_interpfact",
        default=5,
        type=float,
        help="Interpolation factor in ms for spike encoding" "e.g. 1, 2, 3, etc.",
    )

    parser.add_argument(
        "--encoded_data_file_prefix",
        default='encoded-emg',
        type=str,
        help="Interpolation factor in ms for spike encoding" "e.g. 1, 2, 3, etc.",
    )

    parser.add_argument(
        "--learning_algorithm",
        default="critical",
        type=str,
        help="Unsupervised Learning algorithm for reservoir"
             "e.g. critical,stdp,none ",
    )
    parser.add_argument(
        "--stdp_tau",
        default=25,
        type=float,
        help="Tau pre for STDP in ms" "e.g. 10, 20 etc.",
    )
    parser.add_argument(
        "--stdp_apre",
        default=1e-3,
        type=float,
        help="dApre for STDP" "e.g. 1e-4, 1e-2 etc",
    )

    parser.add_argument(
        "--memoryless_flag",
        default=True,
        type=bool,
        help="Forgeting reservoir" "e.g. True,False ",
    )
    parser.add_argument(
        "--online_flag",
        default=True,
        type=bool,
        help="Online learning (Test dataset)" "e.g. True,False",
    )

    parser.add_argument(
        "--topology",
        default="small-world",
        type=str,
        help="Reservoir topology" "e.g. random,custom,small-world ",
    )

    parser.add_argument(
        "--macrocolumnShape",
        default=[2, 5, 1],
        # type=str,
        help=" Macrocolumn Shape ",
    )

    parser.add_argument(
        "--minicolumnShape",
        default=[4, 4, 2],
        # type=str,
        help=" Minicolumn Shape ",
    )

    parser.add_argument(
        "--connection_density",
        default=0.1,
        type=float,
        help="p_max of reservoir connections" "e.g. 0.1, 0.9 ",
    )

    parser.add_argument(
        "--wmax",
        default=1,
        type=float,
        help="Max weight of reservoir connections" "e.g. 1, 0.9 ",
    )
    parser.add_argument(
        "--wmin",
        default=0.01,
        type=float,
        help="Min weight of reservoir connections" "e.g. 0.1, 0.01 ",
    )

    parser.add_argument(
        "--winitmax",
        default=0.25,
        type=float,
        help="Max init weight of reservoir connections" "e.g. 1, 0.9 ",
    )
    parser.add_argument(
        "--winitmin",
        default=0,
        type=float,
        help="Min init weight of reservoir connections" "e.g. 0.1, 0.01 ",
    )
    parser.add_argument(
        "--win",
        default='1 * rand()',
        type=str,
        help="Input weights" "e.g. 0.1, 0.01 ",
    )

    parser.add_argument(
        "--cbf",
        default=1,
        type=float,
        help="CRITICAL target branching factor" "e.g. 1, 0.9 ",
    )
    parser.add_argument(
        "--lr_critical",
        default=0.1,
        type=float,
        help="Min weight of reservoir connections" "e.g. 0.1, 0.01 ",
    )
    parser.add_argument(
        "--excitatoryProb",
        default=0.8,
        type=float,
        help="Proportion of excitatory neurons in reservoir population"
             "e.g. 0.8, 0.9 ",
    )
    parser.add_argument(
        "--adaptiveProb",
        default=1,
        type=float,
        help="Proportion of adaptive neurons in reservoir population" "e.g. 0.1, 0.9 ",
    )
    parser.add_argument(
        "--init_tau",
        default=25,
        type=float,
        help="Minimum value of reservoir neuron leak time constant  ",
    )
    parser.add_argument(
        "--init_tau_dev",
        default=0,
        type=float,
        help="Deviation in the tau(between 0 and 1) ",
    )
    parser.add_argument(
        "--init_thr",
        default=1,
        type=float,
        help="Mean value of reservoir neuron initial threshold ",
    )
    parser.add_argument(
        "--init_thr_dev",
        default=0.5,
        type=float,
        help="Deviation in initial threshold (between 0 and 1) ",
    )
    parser.add_argument(
        "--refractory",
        default=1,
        type=float,
        help="Refractory period of reservoir neurons ",
    )

    parser.add_argument(
        "--tstep",
        default=200,
        type=float,
        help="Readout layer step time in ms" "e.g. 200, 300, etc etc.",
    )

    parser.add_argument(
        "--freeze_time_ms",
        default=0,
        type=float,
        help="Time for which synaptic transmission should be frozen for weight update. (Hardware constraint)",
    )

    parser.add_argument(
        "--tstart",
        default=600,
        type=float,
        help="Time point from which the simulated sub-segment(of length tstep) is used as a feature for readout layer" ">0 (in ms).",
    )

    parser.add_argument(
        "--tlast",
        default=1200,
        type=float,
        help="Time point till which the simulated sub-segment(of length tstep) is used as a feature for readout layer" "e.g. <1800> (in ms).",
    )
    parser.add_argument(
        "--duration",
        default=2000,
        type=float,
        help="Time point till which the simulation has to be run",
    )

    parser.add_argument(
        "--noise",
        default=0,
        type=float,
        help="Spontaneous activity in reservoir",
    )
    parser.add_argument(
        "--seed",
        default=50,
        type=float,
        help="Seed for random number generation",
    )

    parser.add_argument(
        "--input_connection_density",
        default=0.15,
        type=float,
        help="Connection density of input synapses relative to reservoir. (0-1)",
    )

    parser.add_argument('--experiment_name', default='standalone', type=str,
                        help='Name for identifying the experiment'
                               'e.g. plot ')

    parser.add_argument('--target', default='cython', type=str,
                        help='Numpy or Cython target for code gen '
                               'e.g. numpy/cython ')

    parser.add_argument('--dt', default=1, type=float,
                        help='Time step for brian2 simulation'
                               'e.g. 1, 2, etc in ms ')

    parser.add_argument('--fold', default=1, type=float,
                        help='Fold for train/test'
                             'e.g. 1, 2, 3 ')
    parser.add_argument('--classes_dict', default=["rock", "paper", "scissor"], 
                        help='Name of classes')


    parser.add_argument('--path_input_connections', default='inp.txt', 
                        help='Path for matrix with input connections')
    
    parser.add_argument('--path_res_connections', default='Conn_map.txt', 
                        help='Path for matrix with reservoir connections')


    parser.add_argument('--log_file_path', default=None, 
                        help='Path for log file')


    # parser.add_argument('--inputs_list', nargs="?", default=None,  const=inputs_list,
    #                     help='List of inputs to RNN')

    my_args = parser.parse_args()

    return my_args
