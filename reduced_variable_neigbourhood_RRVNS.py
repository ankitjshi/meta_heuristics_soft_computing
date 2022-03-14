#!/usr/bin/env python
# coding: utf-8

import collections
import pandas as pd
import random
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the Database with respective flow and cost matric based on nodes
CAB_10_nodes_flow = pd.read_csv("10_nodes_CAB_flow.csv", delimiter=",", header=None)
CAB_10_nodes_cost = pd.read_csv("10_nodes_CAB_cost.csv", delimiter=",", header=None)

CAB_25_nodes_flow = pd.read_csv("25_nodes_CAB_flow.csv", header=None)
CAB_25_nodes_cost = pd.read_csv("25_nodes_CAB_cost.csv", header=None)

TR_55_nodes_flow = pd.read_csv("55_nodes_TR_flow.csv", header=None)
TR_55_nodes_cost = pd.read_csv("55_nodes_TR_cost.csv", header=None)

TR_81_nodes_flow = pd.read_csv("81_nodes_TR_flow.csv", header=None)
TR_81_nodes_cost = pd.read_csv("81_nodes_TR_cost.csv", header=None)

TR_100_nodes_flow = pd.read_csv("100_nodes_TR_flow.csv", header=None)
TR_100_nodes_cost = pd.read_csv("100_nodes_TR_cost.csv", header=None)

TR_130_nodes_flow = pd.read_csv("130_nodes_TR_flow.csv", header=None)
TR_130_nodes_cost = pd.read_csv("130_nodes_TR_cost.csv", header=None)


# The function will allocate the spokes to hubs based on the least cost
def hub_locator_least_cost(n_nodes, n_hubs, cost_matrix):
    nodes = list(range(1, n_nodes + 1))
    hubs = random.sample(nodes, n_hubs)
    spokes = list(set(nodes) - set(hubs))
    solution = [0] * n_nodes
    for spoke in spokes:
        lst = [[cost_matrix.iloc[spoke - 1][hub - 1], hub] for hub in hubs]
        hub = min(lst, key=lambda x: x[0])[1]
        solution[spoke - 1] = (hub)
    for hub in hubs:
        solution[hub - 1] = hub
    return solution


# NS1 : NEIGHBOURHOOD STRUCTURE 0: swap the current random hub with random non-hub
def neighbourhood_structure0(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    nodes = list(range(1, len(neighbour_struc) + 1))
    spokes = list(set(nodes) - set(hubs))
    select_spoke = random.choice(spokes)
    hub = neighbour_struc[select_spoke - 1]
    for index in range(len(neighbour_struc)):
        if neighbour_struc[index] == hub:
            neighbour_struc[index] = select_spoke
    # print('Neighbourhood Structure0',neighbour_struc)
    return neighbour_struc


# NS2 : NEIGHBOURHOOD STRUCTURE 1: swap the nodes between two hubs
def neighbourhood_structure1(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    hub_spoke = {}
    for hub in list(set(neighbour_struc)):
        hub_spoke[hub] = [i + 1 for i, val in enumerate(neighbour_struc) if val == hub and i + 1 != hub]

    hubs_selected = random.sample(hubs, k=2)
    if len(hub_spoke.get(hubs_selected[0])) < 1 or len(hub_spoke.get(hubs_selected[1])) < 1:
        return neighbour_struc
    spoke_replace1 = random.sample(hub_spoke.get(hubs_selected[0]), k=1)[0]
    spoke_replace2 = random.sample(hub_spoke.get(hubs_selected[1]), k=1)[0]
    neighbour_struc[spoke_replace1 - 1], neighbour_struc[spoke_replace2 - 1] = neighbour_struc[spoke_replace2 - 1], \
                                                                               neighbour_struc[spoke_replace1 - 1]
    # print('Neighbourhood Structure1',neighbour_struc)
    return neighbour_struc


# NS3 : NEIGHBOURHOOD STRUCTURE 2: shift the node from one hub to another
def neighbourhood_structure2(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    nodes = list(range(1, len(neighbour_struc) + 1))
    spokes = list(set(nodes) - set(hubs))
    select_spoke = random.choice(spokes)
    hub = neighbour_struc[select_spoke - 1]
    select_hub = random.choice(hubs)
    while select_hub == hub:
        select_hub = random.choice(hubs)
    neighbour_struc[select_spoke - 1] = select_hub
    # print('Neighbourhood Structure2',neighbour_struc)
    return neighbour_struc


# NS4 : NEIGHBOURHOOD STRUCTURE 3: swap the two nodes between two hubs
def neighbourhood_structure3(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    hub_spoke = {}
    for hub in list(set(neighbour_struc)):
        hub_spoke[hub] = [i + 1 for i, val in enumerate(neighbour_struc) if val == hub and i + 1 != hub]

    hubs_selected = random.sample(hubs, k=2)
    if len(hub_spoke.get(hubs_selected[0])) < 2 or len(hub_spoke.get(hubs_selected[1])) < 2:
        return neighbour_struc

    # for i in range(len(solution)):
    #   if neighbour_struc[i] == hubs_selected[0]:
    #       neighbour_struc[i] == hubs_selected[1]
    #   elif neighbour_struc[i] == hubs_selected[1]:
    #       neighbour_struc[i] == hubs_selected[0]

    spoke_replace1 = random.sample(hub_spoke.get(hubs_selected[0]), k=2)
    spoke_replace2 = random.sample(hub_spoke.get(hubs_selected[1]), k=2)
    neighbour_struc[spoke_replace1[0] - 1], neighbour_struc[spoke_replace2[0] - 1] = neighbour_struc[
                                                                                         spoke_replace2[0] - 1], \
                                                                                     neighbour_struc[
                                                                                         spoke_replace1[1] - 1]
    neighbour_struc[spoke_replace1[1] - 1], neighbour_struc[spoke_replace2[1] - 1] = neighbour_struc[
                                                                                         spoke_replace2[1] - 1], \
                                                                                     neighbour_struc[
                                                                                         spoke_replace1[1] - 1]
    # print('Neighbourhood Structure3',neighbour_struc)
    return neighbour_struc


# Objective/ fitness fucntion to capture the total cost of the traversal
# It normalizes and returns both the total cost as well as the normalised cost
def network_cost(solution, node, flow_matrix, cost_matrix, alpha):
    total_flow_cost = 0
    normalised_flow = 0
    for i in range(0, node):
        for j in range(0, node):
            flow_cost = flow_matrix.iloc[i][j] * (
                    cost_matrix.iloc[i][solution[i] - 1] + alpha * cost_matrix.iloc[solution[i] - 1][
                solution[j] - 1] + cost_matrix.iloc[solution[j] - 1][j])
            total_flow_cost = total_flow_cost + flow_cost
            normalised_flow += flow_matrix[i][j]
    return (total_flow_cost, total_flow_cost / normalised_flow)


#  REDUCED VARIABLE NEIGHBOURHOOD SEARCH - RRVNS works based on 4 neighbouthood strucutres
def reduced_varible_neighbourhood_search(node, n_hub, flow_matrix, cost_matrix, alpha, time_max):
    start = time.time()

    # Initial Solution is generated based on least cost
    initial_solution = hub_locator_least_cost(node, n_hub, cost_matrix)
    initial_solution_cost_total, initial_solution_cost = network_cost(initial_solution, node, flow_matrix, cost_matrix,
                                                                      alpha)

    # Initial Solution assigned to be the best_cost at the beginning
    best_solution = initial_solution.copy()
    best_solution_cost = initial_solution_cost
    best_solution_cost_total = initial_solution_cost_total

    # All the four neighbourhood structures are initialized and traversed one by one
    structures = [neighbourhood_structure0, neighbourhood_structure1, neighbourhood_structure2,
                  neighbourhood_structure3]
    time_contraint = 0
    iteration = 0

    # Iteration runs till stopping criteria is reached (based on the time limit set)
    while time_contraint < time_max:
        st = time.time()

        # For each neighbourhood respective number of solutions NI are generated for evaluation
        NI = [node - n_hub, node, node, n_hub]
        for i in range(0, len(structures)):
            for j in NI:
                structure = structures[i]
                neighbour_solution = structure(best_solution).copy()
                neighbour_cost_total, neighbour_cost = network_cost(neighbour_solution, node, flow_matrix, cost_matrix,
                                                                    alpha)

                # Best cost at any instance is captured and initialsed to initial_Solution for the next iteration to proceed
                if neighbour_cost < best_solution_cost:
                    best_solution = neighbour_solution.copy()
                    best_solution_cost = neighbour_cost
                    best_solution_cost_total = neighbour_cost_total

        et = time.time()
        time_contraint = time_contraint + (et - st)
        iteration += 1
    end = time.time()

    return (
    best_solution_cost, list(set(best_solution)), best_solution, end - start, best_solution_cost_total, iteration)


# Main function takes parameters for the run as per the parameters passed
def run_meta_heuristics(parameters):
    for param in parameters:
        name = param.get("name")
        hub_numbers = param.get("hub_numbers")
        alphas = param.get("alphas")
        nodes_flow = param.get("data")[0]
        nodes_cost = param.get("data")[1]
        node = param.get("nodes")
        time_max = param.get("time_max")
        print("Running RVNS for 10 random initial solutions")
        print("Configuration:", name)

        for n_hub in hub_numbers:
            for alpha in alphas:
                costs = []
                solution_dict = {}
                # Functional loop to run over 10 different initial solutions and iterations
                for i in range(10):
                    # The result will br in this order -- optimal_cost, best_hub_configuration, optimal_hub_spoke_netowrk, time_taken, total_cost, number_of_iterations
                    result = reduced_varible_neighbourhood_search(node, n_hub, nodes_flow, nodes_cost, alpha, time_max)
                    solution_dict[result[0]] = result
                    print(result)
                    costs.append(result[0])
                # Solution is stored in dictionary in order ot get the average cost and print it at the end
                output = collections.OrderedDict(sorted(solution_dict.items()))
                values_view = output.values()
                value_iterator = iter(values_view)
                first_value = next(value_iterator)
                print("==================================================")
                print("Optimum value:", first_value)
                print("Average cost:", sum(costs) / 10)
                print("==================================================")


# Comment and run the as per the requirements
# If only 10 nodes with hub 3 -> 0.2 needs to be run then passed the parametes as below:
# {"name": "CAB 10", "hub_numbers": [3], "alphas": [0.2], "data":[CAB_10_nodes_flow, CAB_10_nodes_cost], "nodes":10, "time_max":50 },
# Uncomment all the parameters to run for larger nodes
# The result will be in this order -- optimal_cost, best_hub_configuration, optimal_hub_spoke_netowrk, time_taken, total_cost, number_of_iterations

parameters = [
    {"name": "CAB 10", "hub_numbers": [3, 5], "alphas": [0.2, 0.8], "data": [CAB_10_nodes_flow, CAB_10_nodes_cost],
     "nodes": 10, "time_max": 30},
    # {"name": "CAB 25", "hub_numbers": [3,5], "alphas": [0.2, 0.8], "data":[CAB_25_nodes_flow, CAB_25_nodes_cost], "nodes":25, "time_max":50 },
    # {"name": "TR 55", "hub_numbers": [3,5], "alphas": [0.2,0.8], "data":[TR_55_nodes_flow, TR_55_nodes_cost], "nodes":55, "time_max":350 },
    # {"name": "TR 81", "hub_numbers": [5,7], "alphas": [0.2,0.8], "data":[TR_81_nodes_flow, TR_81_nodes_cost], "nodes":81, "time_max":1500 },
    # {"name": "TR 100", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data":[TR_100_nodes_flow, TR_100_nodes_cost], "nodes":100, "time_max":3000 },
    # {"name": "TR 130", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data": [TR_130_nodes_flow, TR_130_nodes_cost],"nodes": 130, "time_max":4000}
]

run_meta_heuristics(parameters)
