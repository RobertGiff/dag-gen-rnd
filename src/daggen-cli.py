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
import numpy as np
from tqdm import tqdm

from rnddag import DAG, DAGTaskset
from generator import uunifast_discard, uunifast
from generator import gen_period, gen_execution_times


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

        # DAG generation main loop
        for u_total in np.arange(1.0, u_max+u_step, u_step):
            for set_index in tqdm(range(n_set)):
                logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # create a new taskset
                Gamma = DAGTaskset()

                U_p = []

                # DAG taskset utilization
                U = uunifast_discard(n, u=u_total, nsets=n_set, ulimit=cores)

                # generate periods
                periods = gen_period(period_set, n)
                logging.info(periods)

                for i in range(n):
                    # calculate workload (in us)
                    w = U[set_index][i] * periods[i]

                    # create a new DAG
                    G = DAG(i=i, U=U[set_index][i], T=periods[i], W=w)

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
                    print("RG MODIFICATIONS")
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
                        print(f"workload chosen: {random_subdir}")

                        random_subdir_path = os.path.join(profile_path, random_subdir)

                        # Construct the full path to the file
                        #file_path = os.path.join(random_subdir_path, "1048575_1440/wcet.txt") # ref of 20/20
                        file_path = os.path.join(random_subdir_path, "31_360/wcet.txt")  # ref of 5/5

                        # Open the file
                        try:
                            with open(file_path, 'r') as file:
                                wcet = float(file.readline().strip())
                                wcet *= 1_000_000_000 # convert from s to ns
                                print(f"ref wcet: {wcet}")
                        except FileNotFoundError:
                            print(f"The file '{file_path}' does not exist.")
                        except Exception as e:
                            print(f"An error occurred: {e}")

                        # Generate our own WCET using a random workload
                        data['C'] = wcet  # For example, increment 'C' by 10
                        data['type'] = random_subdir
                        sum_wcet += wcet

                    # Now calculate what the period is going to be
                    print(f"sum wcet: {sum_wcet}")
                    G.G.graph['W'] = sum_wcet

                    util = u_p 
                    print(f"Target util for this task graph: {util}")

                    period = sum_wcet / util
                    # Now round to the nearest multiple of 2, in seconds, to set up a harmonic period value
                    #period = round_up_to_nearest_2_seconds(period)
                    #period = round_to_nearest_second(period)
                    #period = round_to_nearest_2_seconds(period)
                    #period = round_to_nearest_5_seconds(period)
                    #period = round_to_nearest_100ms(period)
                    period = round_to_nearest_power_of_2(period)
                    G.G.graph['T'] = int(period)

                    # Now re-update the util to account for this rounding
                    G.G.graph['U'] = sum_wcet / period
                    print(f"New period for this task graph: {period}")
                    print(f"New util for this task graph: {G.G.graph['U']}")
                    print(graph.nodes.data())

                    # save the graph
                    if config["misc"]["save_to_file"]:
                        G.save(basefolder="./data/data-multi-m{}-u{:.1f}/{}/".format(cores, u_total, set_index))

                    # (optional) plot the graph
                    # G.plot()

                logging.info("Total U:", sum(U_p), U_p)
                logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                logging.info("")
