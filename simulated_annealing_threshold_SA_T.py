# -*- coding: utf-8 -*-
"""Simulated_Annealing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y_YHAcRZNmoI_DJSLI4fVwOZr9LKfPQM
"""
import pandas as pd
import random
import time
from collections import Counter

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


# Objective/ fitness function to capture the total cost of the traversal
# It normalizes and returns both the total cost as well as the normalised cost
def network_cost(initial_solution, flow_matrix, cost_matrix, alpha=0.2):
    number_nodes = cost_matrix.shape[0]
    cost = 0
    flow = 0
    for node_1 in range(number_nodes):
        for node_2 in range(number_nodes):
            cost += flow_matrix[node_1][node_2] * (cost_matrix[node_1][initial_solution[node_1] - 1] +
                                                   alpha * cost_matrix[initial_solution[node_1] - 1][
                                                       initial_solution[node_2] - 1] +
                                                   cost_matrix[initial_solution[node_2] - 1][node_2])
            flow += flow_matrix[node_1][node_2]

    return (cost / flow, cost)


# Calculates the NS1 : randomly convert a spoke to hub and assign the previous single hub to another cluster as spoke.
def type_ns_1(initial_solution, nodes):
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(nodes)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]
    num_spokes = dict(Counter(solution))
    hub_single_spoke = list(num_spokes.keys())[list(num_spokes.values()).index(1)]
    random_spoke = random.choice(spokes)
    solution[hub_single_spoke - 1] = random_spoke
    solution[random_spoke - 1] = random_spoke
    return solution


# Calculates the NS2 : randomly swap the spoke to hub of another cluster
def type_ns_2(initial_solution, nodes):
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(nodes)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]
    random_hub = random.choice(hubs)
    random_spoke = random.choice(spokes)
    solution = [random_spoke if i == random_hub else i for i in solution]
    solution[random_spoke - 1] = random_spoke
    return solution


# Calculates the NS3 : randomly swap the spoke from one hub to the another nearest hub
def type_ns_3(initial_solution, cost, nodes):
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(nodes)]
    hubs = list(set(solution))
    p = len(hubs)
    spokes = [node for node in nodes if node not in hubs]
    random_spoke = random.choice(spokes)
    choice_hubs = list(set(hubs) - {solution[random_spoke - 1]})
    hub_spoke_cost = {hub: cost[random_spoke - 1][hub - 1] for hub in choice_hubs}
    hub_spoke_cost = dict(sorted(hub_spoke_cost.items(), key=lambda item: item[1]))
    if p <= 4:
        top_hubs = list(hub_spoke_cost.keys())[:(p - 1)]
    else:
        top_hubs = list(hub_spoke_cost.keys())[:4]
    random_hub = random.choice(top_hubs)
    solution[random_spoke - 1] = random_hub

    return solution


# Calculates the NS4 : randomly swap the hub to the spoke of same cluster
def type_ns_4(initial_solution, cost, nodes):
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(nodes)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]
    random_hub = random.choice(hubs)
    spokes_of_hub = [i for i in spokes if solution[i - 1] == random_hub]
    random_spoke = random.choice(spokes_of_hub)
    solution = [random_spoke if i == random_hub else i for i in solution]
    return solution


# The function will allocate the spokes to hubs based on the least cost
def least_cost_initial_solution(cost_matrix, number_hubs):
    number_nodes = cost_matrix.shape[0]
    nodes = range(1, number_nodes + 1)
    random.sample(nodes, number_hubs)
    hubs = random.sample(nodes, number_hubs)
    spokes = [node for node in nodes if node not in hubs]
    initial_solution = [0] * number_nodes

    for node in nodes:
        if node in hubs:
            initial_solution[node - 1] = node
        else:
            hub_spoke_cost = {hub: cost_matrix[node - 1][hub - 1] for hub in hubs}
            initial_solution[node - 1] = min(hub_spoke_cost, key=hub_spoke_cost.get)
    return initial_solution


# Alist for NS3 and NS4 as they have 50 percent probbaility of selection
ns_type_3_4 = [type_ns_3, type_ns_4]


# Function to provide a local search replacing hub with spoke
def neighbourhood_structure_1(array, spoke):
    n_array = array.copy()
    hub = n_array[spoke - 1]
    for i in range(len(n_array)):
        if n_array[i] == hub:
            n_array[i] = spoke
    return n_array


# Function to provide a local search for each spoke replace with hub
def neighbourhood_structure_1_steepest(array, w, c, alpha):
    best_neighbour = array.copy()
    best_neighbour_cost, best_neighbour_tot_cost = network_cost(array, w, c, alpha)
    spokes = [i for i in range(1, len(array) + 1) if i not in array]
    for s in spokes:
        neighbour = neighbourhood_structure_1(array, s)
        neighbour_cost, neighbour_tot_cost = network_cost(neighbour, w, c, alpha)
        if neighbour_cost < best_neighbour_cost:
            best_neighbour = neighbour.copy()
            best_neighbour_cost = neighbour_cost
            best_neighbour_tot_cost = neighbour_tot_cost
    return best_neighbour, best_neighbour_cost, best_neighbour_tot_cost


#  Threshold based SA - SA-T - works on threshold and reduces based on gamma
def simulated_annealing_threshold(nodes, p_hub, flow_matrix, cost_matrix, alpha, iter):
    start = time.time()

    # Runs the iteration for 10 new initial solutions
    for i in range(iter):

        # Initial solution is generated based on the least cost matrix
        initial_solution = least_cost_initial_solution(cost_matrix, p_hub)

        # Initial solution cost is calculated
        initial_cost, initial_tot_cost = network_cost(initial_solution, flow_matrix, cost_matrix, alpha=alpha)

        # Initial cost is then assigned to the best solution and cost at the beginning
        current_solution = initial_solution.copy()
        current_cost = initial_cost
        best_solution = initial_solution.copy()
        best_cost = initial_cost

        # paramters THo -- Initial treshold set 0.01 * cost
        # total number of iterations k set to default 120
        # M_threshold set to 0.9
        # threshold keeps decreasing by factor gamma startes with inital_threshold
        # M_threshold sets at nodes*hub/10
        intitial_threshold = 0.01 * initial_cost
        K_tot_iter = 120
        gama_factor = 0.9
        M_threshold = (nodes * p_hub) / 10
        threshold = intitial_threshold

        # First iteration runs till k
        iter_j = 1
        while iter_j < K_tot_iter:
            iter_i = 1

            # Second iteration runs till M_threshold
            while iter_i <= M_threshold:

                # It goest and cheks for the criteria
                # Criteria 1 : if there exists cluster with just one hub and no spokes; else move to next condition
                # Criteria 1 Selects NS1
                if (1 in dict(Counter(current_solution)).values()):
                    neighbour = type_ns_1(current_solution, nodes)
                    neighbour_cost, neighbour_tot_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha=alpha)
                # Criteria 2 : if iteration is equal to M_threshold perform Action; else select either 〖NS〗_3/〖NS〗_4
                # Criteria 2 Selects NS2
                elif iter_i == M_threshold:
                    neighbour = type_ns_2(current_solution, nodes)
                    neighbour_cost, neighbour_tot_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha=alpha)
                # Criteria 3 : selection for type NS3 and NS4 is 50-50 percent
                else:
                    neighbour = random.choice(ns_type_3_4)(current_solution, cost_matrix, nodes)
                    neighbour_cost, neighbour_tot_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha=alpha)

                # Delta E or difference calculated best on the neigbour_cost and current_cost
                delta_E = neighbour_cost - current_cost

                # if delta is less than threshold then simply perform the replacement and update the best solutions
                if delta_E <= threshold:
                    current_solution = neighbour.copy()
                    current_cost = neighbour_cost
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost
                iter_i += 1

            # Every iteration decreases the threshold by factor M_threshold
            threshold = gama_factor * threshold
            iter_j += 1

        # At the end local search is performed on the best solution to find the optimal soltuion and configuration
        best_solution, best_cost, best_cost_tot = neighbourhood_structure_1_steepest(best_solution, flow_matrix,
                                                                                     cost_matrix, alpha=alpha)
        end = time.time()

        # prints output as optimal_normalized_cost, optimal_configuration, optimal_hubs_selection, total_cost, time_taken
        print(best_cost, best_solution, list(set(best_solution)), best_cost_tot, end - start)
    return (best_cost, best_solution, list(set(best_solution)), best_cost_tot, end - start)


# Main Function to run the heuristic based on the parameters passed
def run_meta_heuristics(parameters):
    for param in parameters:
        name = param.get("name")
        hub_numbers = param.get("hub_numbers")
        alphas = param.get("alphas")
        nodes_flow = param.get("data")[0]
        nodes_cost = param.get("data")[1]
        node = param.get("nodes")
        iter = 20
        print("Running Simulated Annealing for nodes", node)

        for n_hub in hub_numbers:
            for alpha in alphas:
                print("Configuration:", name, "hub:", n_hub, "->", alpha)

                # Runs the algorithm based on the hub and alpha passed as parameters
                simulated_annealing_threshold(node, n_hub, nodes_flow, nodes_cost, alpha, iter)
                print("==================================================")


# Comment and run the as per the requirements
# If only 10 nodes with hub 3 -> 0.2 needs to be run then passed the parametes as below:
# {"name": "CAB 10", "hub_numbers": [3], "alphas": [0.2], "data":[CAB_10_nodes_flow, CAB_10_nodes_cost], "nodes":10},
# Uncomment all the parameters to run for larger nodes
# prints result in order --> optimal_normalized_cost, optimal_configuration, optimal_hubs_selection, total_cost, time_taken

parameters = [
    {"name": "CAB 10", "hub_numbers": [3, 5], "alphas": [0.2, 0.8], "data": [CAB_10_nodes_flow, CAB_10_nodes_cost],
     "nodes": 10},
    # {"name": "CAB 25", "hub_numbers": [3,5], "alphas": [0.2, 0.8], "data":[CAB_25_nodes_flow, CAB_25_nodes_cost], "nodes":25 },
    # {"name": "TR 55", "hub_numbers": [3,5], "alphas": [0.2,0.8], "data":[TR_55_nodes_flow, TR_55_nodes_cost], "nodes":55 },
    # {"name": "TR 81", "hub_numbers": [5,7], "alphas": [0.2,0.8], "data":[TR_81_nodes_flow, TR_81_nodes_cost], "nodes":81},
    # {"name": "TR 100", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data":[TR_100_nodes_flow, TR_100_nodes_cost], "nodes":100  },
    # {"name": "TR 130", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data": [TR_130_nodes_flow, TR_130_nodes_cost],"nodes": 130 }
]

run_meta_heuristics(parameters)
