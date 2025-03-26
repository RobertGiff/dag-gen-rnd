#!/usr/bin/python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Randomized Multi-DAG Task Generator
# Xiaotian Dai
# Real-Time Systems Group
# University of York, UK
# -------------------------------------------------------------------------------

import os, sys, logging, getopt, time, json
import networkx as nx
import random
import time
import numpy as np
from tqdm import tqdm

from rnddag import DAG, DAGTaskset
from generator import uunifast_discard, uunifast
from generator import gen_period, gen_execution_times
from utility import compute_hyper_period

def parse_configuration(config_path):
    try:
        with open(config_path, "r") as config_file:
            config_json = json.load(config_file)
    except:
        raise EnvironmentError("Unable to open %s" % (config_path))

    return config_json

def round_to_nearest_power_of_2(value_ns):
    """
    Rounds the given value in nanoseconds to the nearest power of 2.

    Args:
        value_ns (int): The period value in nanoseconds.

    Returns:
        int: The rounded value in nanoseconds, which is a power of 2.
    """
    import math
    # Calculate the nearest power of 2
    power = round(math.log2(value_ns))
    return int(2 ** power)

def round_to_nearest_2_seconds(ns):
    # Convert 2 seconds to nanoseconds
    two_seconds_ns = 2 * 1_000_000_000

    # Calculate the remainder when divided by two seconds in nanoseconds
    remainder = ns % two_seconds_ns

    # Determine whether to round up or down to the nearest multiple of 2 seconds
    if remainder >= (two_seconds_ns / 2):
        # If remainder is greater than or equal to half of 2 seconds, round up
        rounded_ns = ns + (two_seconds_ns - remainder)
    else:
        # Otherwise, round down
        rounded_ns = ns - remainder

    return rounded_ns

def round_to_nearest_5_seconds(ns):
    # Convert 2 seconds to nanoseconds
    five_seconds_ns = 5 * 1_000_000_000

    # Calculate the remainder when divided by two seconds in nanoseconds
    remainder = ns % five_seconds_ns

    # Determine whether to round up or down to the nearest multiple of 2 seconds
    if remainder >= (five_seconds_ns / 2):
        # If remainder is greater than or equal to half of 2 seconds, round up
        rounded_ns = ns + (five_seconds_ns - remainder)
    else:
        # Otherwise, round down
        rounded_ns = ns - remainder

    return rounded_ns

def round_to_nearest_100ms(ns):
    hundred_ms_ns = 100_000_000
    remainder = ns % hundred_ms_ns

    if remainder >= (hundred_ms_ns / 2):
        rounded_ns = ns + (hundred_ms_ns - remainder)
    else:
        rounded_ns = ns - remainder

    if rounded_ns == 0:
        rounded_ns = hundred_ms_ns

    return rounded_ns

def round_to_nearest_second(ns):
    second_ns = 1_000_000_000
    remainder = ns % second_ns

    if remainder >= (second_ns / 2):
        rounded_ns = ns + (second_ns - remainder)
    else:
        rounded_ns = ns - remainder

    return rounded_ns

def round_up_to_nearest_2_seconds(ns):
    # Convert 2 seconds to nanoseconds
    #two_seconds_ns = 2 * 1_000_000_000
    two_seconds_ns = 1_000_000_000

    # Calculate the remainder when divided by two seconds in nanoseconds
    remainder = ns % two_seconds_ns

    # If there's any remainder, round up to the next multiple
    if remainder != 0:
        rounded_ns = ns + (two_seconds_ns - remainder)
    else:
        rounded_ns = ns

    return rounded_ns


def print_usage_info():
    logging.info("[Usage] python3 daggen-cli.py --config config_file")


def calculate_critical_path_length(G):
    # Make sure we're working with a directed graph
    if not G.is_directed():
        raise ValueError("Graph must be directed")
    
    # Get the execution time for each node
    execution_times = nx.get_node_attributes(G, 'C')
    
    # Initialize earliest completion time for all nodes
    earliest_completion = {}
    
    # Topological sort the graph
    for node in nx.topological_sort(G):
        # Initialize with the node's execution time
        earliest_completion[node] = execution_times.get(node, 0)
        
        # Check all predecessors and update if necessary
        max_pred_path = 0
        for pred in G.predecessors(node):
            if pred in earliest_completion:
                path_length = earliest_completion[pred]
                max_pred_path = max(max_pred_path, path_length)
        
        # Add the maximum predecessor path length to the current node's execution time
        earliest_completion[node] += max_pred_path
    
    # The critical path length is the maximum value in earliest_completion
    if earliest_completion:
        return max(earliest_completion.values())
    else:
        return 0


if __name__ == "__main__":
    ############################################################################
    # Initialize directories
    ############################################################################
    src_path = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.abspath(os.path.join(src_path, os.pardir))

    data_path = os.path.join(base_path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logs_path = os.path.join(base_path, "logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    ############################################################################
    # Parse cmd arguments
    ############################################################################
    config_path = os.path.join(base_path, "config.json")
    directory = None
    load_jobs = False
    evaluate = False
    train = False

    try:
        short_flags = "hc:d:e"
        long_flags = ["help", "config=", "directory=", "evaluate"]
        opts, args = getopt.getopt(sys.argv[1:], short_flags, long_flags)
    except getopt.GetoptError as err:
        logging.error(err)
        print_usage_info()
        sys.exit(2)

    logging.info("Options:", opts)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage_info()
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-d", "--directory"):
            directory = arg
            load_jobs = True
        elif opt in ("-e", "--evaluate"):
            evaluate = True
        else:
            raise ValueError("Unknown (opt, arg): (%s, %s)" % (opt, arg))

    # load configuration
    config = parse_configuration(config_path)

    logging.info("Configurations:", config)

    ############################################################################
    # load generator basic configuration
    ############################################################################
    # load and set random seed
    random.seed(config["misc"]["rnd_seed"])

    # single- or multi-dag
    multi_dag = config["misc"]["multi-DAG"]

    # DAG config
    dag_config = config["dag_config"]

    ############################################################################
    # I. single DAG generation
    ############################################################################
    if not multi_dag:
        n = config["single_task"]["set_number"]
        w = config["single_task"]["workload"]

        for i in tqdm(range(n)):
            # create a new DAG
            G = DAG(i=i, U=-1, T=-1, W=w)
            G.gen_rnd(parallelism=dag_config["parallelism"],
                      layer_num_min=dag_config["layer_num_min"],
                      layer_num_max=dag_config["layer_num_max"],
                      connect_prob=dag_config["connect_prob"])

            # generate sub-DAG execution times
            n_nodes = G.get_number_of_nodes()
            dummy = config["misc"]["dummy_source_and_sink"]
            c_ = gen_execution_times(n_nodes, w, round_c=True, dummy=dummy)
            nx.set_node_attributes(G.get_graph(), c_, 'C')

            # set execution times on edges
            w_e = {}
            for e in G.get_graph().edges():
                ccc = c_[e[0]]
                w_e[e] = ccc

            nx.set_edge_attributes(G.get_graph(), w_e, 'label')

            # print internal data
            if config["misc"]["print_DAG"]:
                G.print_data()

            # save graph
            if config["misc"]["save_to_file"]:
                G.save(basefolder="./data/")

    ############################################################################
    # II. multi-DAG generation
    ############################################################################
    else:
        # set of tasksets
        n_set = config["multi_task"]["set_number"]

        # max utilization
        u_max = config["multi_task"]["utilization"]
        u_step = config["multi_task"]["util_step"]

        # task number
        n = config["multi_task"]["task_number_per_set"]

        # number of cores
        cores = config["misc"]["cores"]

        # Load DAG period set (in us)
        period_set = config["multi_task"]["periods"]
        period_set = [(x) for x in period_set]

        # Track how many of each util we have generated
        current_index = {}
        target_utils = []
        for u_total in np.arange(1.4, u_max+u_step, u_step):
            u_total = round(u_total, 1)
            target_utils.append(u_total)
            current_index[u_total] = 0

        # Summary stats
        num_tasksets_generated = 0
        num_tasksets_skipped = 0
        num_tasksets_unschedulable = 0
        num_taskset_saved = 0
        start_time = time.time()

        # DAG generation main loop
        #for u_total in np.arange(1.0, u_max+u_step, u_step):
        for u_total in target_utils:
            print(f"Trying to populate utilization: {u_total}")
            while current_index[u_total] < n_set:
                num_tasksets_generated += 1
                logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # create a new taskset
                taskset = []

                U_p = []

                # DAG taskset utilization
                U = uunifast_discard(n, u=u_total, nsets=1, ulimit=cores)

                # generate periods
                periods = gen_period(period_set, n)
                logging.info(periods)

                skip_this_taskset = False

                for i in range(n):
                    # calculate workload (in us)
                    w = U[0][i] * periods[i]

                    # create a new DAG
                    G = DAG(i=i, U=U[0][i], T=periods[i], W=w)

                    # generate nodes in the DAG
                    # G.gen_nfj()
                    G.gen_rnd(parallelism=dag_config["parallelism"],
                              layer_num_min=dag_config["layer_num_min"],
                              layer_num_max=dag_config["layer_num_max"],
                              connect_prob=dag_config["connect_prob"])

                    # generate sub-DAG execution times
                    n_nodes = G.get_number_of_nodes()
                    dummy = config["misc"]["dummy_source_and_sink"]
                    c_ = gen_execution_times(n_nodes, w, round_c=True, dummy=dummy)
                    nx.set_node_attributes(G.get_graph(), c_, 'C')

                    # calculate actual workload and utilization
                    w_p = 0
                    for item in c_.items():
                        w_p = w_p + item[1]

                    u_p = w_p / periods[i]
                    U_p.append(u_p)

                    # print("Task {}: U = {}, T = {}, W = {}>>".format(i, U[0][i], periods[i], w))
                    # print("w = {}, w' = {}, diff = {}".format(w, w_p, (w_p - w) / w * 100))

                    # set execution times on edges
                    w_e = {}
                    for e in G.get_graph().edges():
                        ccc = c_[e[0]]
                        w_e[e] = ccc

                    nx.set_edge_attributes(G.get_graph(), w_e, 'label')

                    # print internal data
                    if config["misc"]["print_DAG"]:
                        G.print_data()
                        logging.info("")

                    # RG Modifications
                    profile_path = os.path.join(base_path, "profiles")
                    # The task graph is saved in the DAG named G
                    #print("RG MODIFICATIONS")
                    graph = G.get_graph()
                    sum_wcet = 0.0
                    for node, data in graph.nodes(data=True):
                        assert 'C' in data

                        # Randomly select a workload to use
                        # Get all subdirectories in profile_path
                        subdirs = [d for d in os.listdir(profile_path) if os.path.isdir(os.path.join(profile_path, d))]
                        assert subdirs, "No subdirectories found in the given path."
                        # Select one of the subdirectories at random
                        random_subdir = random.choice(subdirs)
                        #print(f"workload chosen: {random_subdir}")

                        random_subdir_path = os.path.join(profile_path, random_subdir)

                        # Construct the full path to the file
                        #file_path = os.path.join(random_subdir_path, "1048575_1440/wcet.txt") # ref of 20/20
                        file_path = os.path.join(random_subdir_path, "31_360/wcet.txt")  # ref of 5/5

                        # Open the file
                        try:
                            with open(file_path, 'r') as file:
                                wcet = float(file.readline().strip())
                                wcet *= 1_000_000_000 # convert from s to ns
                                #print(f"ref wcet: {wcet}")
                        except FileNotFoundError:
                            print(f"The file '{file_path}' does not exist.")
                        except Exception as e:
                            print(f"An error occurred: {e}")

                        # Generate our own WCET using a random workload
                        data['C'] = wcet  # For example, increment 'C' by 10
                        data['type'] = random_subdir
                        sum_wcet += wcet

                    # Now calculate what the period is going to be
                    #print(f"sum wcet: {sum_wcet}")
                    G.G.graph['W'] = sum_wcet

                    util = u_p 
                    #print(f"Target util for this task graph: {util}")

                    period = sum_wcet / util
                    # Now round to the nearest multiple of 2, in seconds, to set up a harmonic period value
                    #period = round_up_to_nearest_2_seconds(period)
                    #period = round_to_nearest_second(period)
                    #period = round_to_nearest_2_seconds(period)
                    #period = round_to_nearest_5_seconds(period)
                    #period = round_to_nearest_100ms(period)
                    period = round_to_nearest_power_of_2(period)
                    G.G.graph['T'] = str(int(period))

                    crit_path = calculate_critical_path_length(G.get_graph())
                    if crit_path > period:
                        skip_this_taskset = True
                        #print(f"Critical path longer than period, retry")
                        num_tasksets_unschedulable += 1
                        break
                        
                    # Now re-update the util to account for this rounding
                    G.G.graph['U'] = sum_wcet / period
                    #print(f"New period for this task graph: {period}")
                    #print(f"New util for this task graph: {G.G.graph['U']}")
                    #print(graph.nodes.data())

                    taskset.append(G)

                periods = []
                if periods:
                    for task in taskset:
                        periods.append(int(task.G.graph['T']))
                    smallest_period = min(periods)
                    hyperperiod = compute_hyper_period(periods)

                    if hyperperiod / smallest_period > 50:
                        num_tasksets_skipped += 1
                        skip_this_taskset = True

                if config["misc"]["save_to_file"] and not skip_this_taskset:
                    u_actual = 0
                    for task in taskset:
                        u_actual += task.G.graph['U']
                    u_actual = round(u_actual, 1)
                    
                    if u_actual in target_utils and current_index[u_actual] < n_set:
                        #print(f"SAVING this taskset with utilization {u_actual} to index {current_index[u_actual]}")
                        for task in taskset:
                            task.save(basefolder="./data/data-multi-m{}-u{:.1f}/{}/".format(cores, u_actual, current_index[u_actual]))
                        current_index[u_actual] += 1
                        num_taskset_saved += 1
                    else:
                        #print(f"SKIPPING this taskset with utilization {u_actual}")
                        num_tasksets_skipped += 1

                    # (optional) plot the graph
                    # G.plot()

                logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                logging.info("")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        print(f"Num tasksets skipped {num_tasksets_skipped/num_tasksets_generated:.2f}")
        print(f"Num tasksets unschedulable {num_tasksets_unschedulable/num_tasksets_generated:.2f}")
        print(f"Num tasksets saved {num_taskset_saved/num_tasksets_generated:.2f}")