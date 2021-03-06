# -*- coding: utf-8 -*-
"""Particle_Swarm_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vFQhZwzBQfLgKUbfRaKxkbQYaeBNIx5Z
"""
import collections
import pandas as pd
import random
import time
from scipy.stats import rankdata


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


def hub_locator_least_cost(cost_matrix, solution1):
    n_nodes = cost_matrix.shape[0]
    nodes = list(range(1, n_nodes + 1))
    hubs = set(solution1)
    spokes = list(set(nodes) - set(hubs))
    solution = [0] * n_nodes
    for spoke in spokes:
        lst = [[cost_matrix.iloc[spoke - 1][hub - 1], hub] for hub in hubs]
        hub = min(lst, key=lambda x: x[0])[1]
        solution[spoke - 1] = (hub)
    for hub in hubs:
        solution[hub-1]=hub
    return solution

# Objective/ fitness function to capture the total cost of the traversal
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
    return (total_flow_cost/normalised_flow)


# NEIGHBOURHOOD STRUCTURE 0: swap the current random hub with random non-hub***1-s2.0-S0305054817302538
def neighbourhood_structure0(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    nodes = list(range(1, len(neighbour_struc)+1))
    spokes = list(set(nodes) - set(hubs))
    select_spoke = random.choice(spokes)
    hub = neighbour_struc[select_spoke-1]
    for index in range(len(neighbour_struc)):
        if neighbour_struc[index]==hub:
            neighbour_struc[index] = select_spoke
    # print('Neighbourhood Structure0',neighbour_struc)
    return neighbour_struc

# Applies diversification if needed
def apply_diversification(neighbour):
    diversified_list = []
    diversified_list.append(neighbour)
    for i in range(1,5):
        neighbour_dup = neighbourhood_structure0(neighbour)
        if neighbour_dup not in diversified_list:
            diversified_list.append(neighbour_dup)
    # print(diversified_list)
    return diversified_list

# Generates solutions based on the ranking and arrangement of the matrices 2*n
def solution_generator(n_hub,Xparticle_1, Xparticle_2):
    array = [Xparticle_1, Xparticle_2]
    array_1 = array.copy()
    array_1[0] = rankdata(array[0], method='ordinal').tolist()

    array_1[1] = [int(i) for i in array[1]]
    hubs = array_1[0][:n_hub]

    solution = [hubs[i - 1] for i in array_1[1]]
    # repairing
    for hub in hubs:
        solution[hub - 1] = hub
    return solution

# initializes population generatesd swarm of random partciles with continuous vlaues ranges as per the constants
# X_particle --> position vector
# V_particle --> velocity vector
# returns the list of list of position and velocity per particle with populaiton of swarm
def generate_particles_swarm_inizialize(Dimensions,XparticleSwarm,VparticleSwarm,Vmin,Vmax,X_1_max,X_2_min,X_2_max):
    X_particle_1 = []
    X_particle_2 = []
    V_particle_1 = []
    V_particle_2 = []
    for i in range(Dimensions):
        X_particle_1.append(X_1_max * random.uniform(0, 1))
        X_particle_2.append(random.uniform(X_2_min, X_2_max))
        V_particle_1.append(Vmin + (Vmax - Vmin) * random.uniform(0, 1))
        rand = random.uniform(0, 1)
        if rand > 0.5:
            V_particle_2.append(Vmin + (Vmax - Vmin) * rand)
        else:
            V_particle_2.append(-1 * (Vmin + (Vmax - Vmin) * rand))

    XparticleSwarm.append([X_particle_1, X_particle_2])
    VparticleSwarm.append([V_particle_1, V_particle_2])
    return [XparticleSwarm, VparticleSwarm]

# Partcile_swarm_discrete heuristic to generate and update continuous particles  and converts to discrete posiitons
# to obtain optimal solutions
def particle_swarm_heuristic(node, n_hub, flow_matrix, cost_matrix, alpha, n_iter):
  start = time.time()

  # initializes the particle constants like swarm size
  # dimensions are number of nodes --> length of particles inside the swarms
  # initialzes the global and particle best to update it at the later stages
  SwarmSize = 25
  Dimensions=node
  iter=0
  global_best_cost=10000
  particle_best_cost=100000

  # velocity parametes for continuous value ranges min to max
  Vmin = 0
  Vmax = n_hub * node * 0.1

  # position parametes for continuous value ranges min to max
  # 1 and 2 represent the first and the second rows of the discrete formation array
  X_1_min = 0
  X_1_max = n_hub * node * 0.1

  X_2_min = 1
  X_2_max = n_hub + 1 - 0.01

  # update Velocity and positions  constants
  r1 = random.uniform(0, 1)
  r2 = random.uniform(0, 1)
  C1 = 2
  C2 = 2
  W = 1

  XparticleSwarm=[]
  VparticleSwarm=[]

  # generates swarm of length SwarmSize: consisiting particles with position and velocity vector
  for i in range(SwarmSize):
      Swarm = generate_particles_swarm_inizialize(Dimensions,XparticleSwarm,VparticleSwarm,Vmin,Vmax,X_1_max,X_2_min,X_2_max)
      XparticleSwarm = Swarm[0]
      VparticleSwarm = Swarm[1]

  # continuous iteration till stopping condition  : which is 1000 in this case
  while iter<n_iter:
    for i in range(SwarmSize):
      X_particle_1 = XparticleSwarm[i][0]
      X_particle_2 = XparticleSwarm[i][1]
      V_particle_1 = VparticleSwarm[i][0]
      V_particle_2 = VparticleSwarm[i][1]

      # After swarm generation the particles are now converted to its discrete form
      discrete_particle= [solution_generator(n_hub,X_particle_1, X_particle_2)]

      # Another appraches like least hub-spoke allocations and idversification based on neighbours generation
      # were also practised to see the improvement in results
      # discrete_particle_solution = [hub_locator_least_cost(cost_matrix,discrete_particle[0])]
      # diversified_swarm_population = apply_diversification(particles[0])

      for particle in discrete_particle:
          # Particle cost is calculated
          particle_cost = network_cost(particle, node, flow_matrix, cost_matrix, alpha)
          # print("Solution generated", diversified_swarm_population[0],"particle_cost:",particle_cost)

          # IF Particle cost is minimial, replacement for current and best is made
          if particle_cost < particle_best_cost:
              particle_best_cost = particle_cost
              particle_best_solution = particle
              local_best_particle = [X_particle_1, X_particle_2]

          if particle_cost < global_best_cost:
              global_best_cost = particle_cost
              global_best_solution = particle
              global_best_particle = [X_particle_1, X_particle_2]

    # velocity and position updation process begins for the next values in the swarm
    for i in range(SwarmSize-1):
      X_particle_1_next = XparticleSwarm[i + 1][0]
      X_particle_2_next = XparticleSwarm[i + 1][1]
      V_particle_1_next = VparticleSwarm[i + 1][0]
      V_particle_2_next = VparticleSwarm[i + 1][1]
      V_particle_1 = VparticleSwarm[i][0]
      V_particle_2 = VparticleSwarm[i][1]
      X_particle_1 = XparticleSwarm[i][0]
      X_particle_2 = XparticleSwarm[i][1]

      for d in range(Dimensions):

        # Velocity updates for two rows of 2*n matrix 1 represents the vector associated to first row
        # and 2 represents the vector associated to second row
        V_particle_1_next[d] = W * V_particle_1[d] + C1 * r1 * (local_best_particle[0][d]- X_particle_1[d])\
                + C2 * r2 * (global_best_particle[0][d] - X_particle_1[d])
        V_particle_2_next[d] = W * V_particle_2[d] + C1 * r1 * (local_best_particle[1][d]- X_particle_2[d])\
                + C2 * r2 * (global_best_particle[1][d] - X_particle_2[d])

        # After the velocity update the cap to velocities is made
        # so that they dont leave and fly to higher values
        if V_particle_1_next[d] > Vmax:
            V_particle_1_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
        if V_particle_1_next[d] < Vmin:
            V_particle_1_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)

        # updates for positive and negative vlaues of velocities
        if V_particle_2_next[d] > 0:
          if V_particle_2_next[d] > Vmax:
              V_particle_2_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
          if V_particle_2_next[d] < Vmin:
              V_particle_2_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
        else:
          if abs(V_particle_2_next[d]) > Vmax:
              V_particle_2_next[d] = -1*(Vmin + (Vmax - Vmin) * random.uniform(0, 1))
          if abs(V_particle_2_next[d]) < Vmin:
              V_particle_2_next[d] = -1*(Vmin + (Vmax - Vmin) * random.uniform(0, 1))

        # Update in position for 1 and 2 again signifies first and second row ofthe matrix
        X_particle_1_next[d] = X_particle_1[d] + V_particle_1_next[d]
        X_particle_2_next[d] = X_particle_2[d] + V_particle_2_next[d]

        # similar cap for postion is made
        # so that they dont leave and fly to higher value
        if X_particle_1_next[d] > X_1_max:
          X_particle_1_next[d] = X_1_max*random.uniform(0, 1)
        if X_particle_1_next[d] < X_1_min:
          X_particle_1_next[d] = X_1_max*random.uniform(0, 1)

        if X_particle_2_next[d] > X_2_max:
          X_particle_2_next[d] = random.uniform(X_2_min, X_2_max)
        if X_particle_2_next[d] < X_2_min:
          X_particle_2_next[d] = random.uniform(X_2_min, X_2_max)

        # After the updates are performed, the new ones are replaced in the swarm with previous values
        # in order to be picked in next iteration
        XparticleSwarm[i+1] = [X_particle_1_next, X_particle_2_next]
        VparticleSwarm[i+1] = [V_particle_1_next, V_particle_2_next]
    # print("global_best_solution",global_best_solution,"global_best_cost",global_best_cost)
    # print("Updating new swarm")
    # print("<==============================================================================================>")

    iter += 1
  end = time.time()
  return (global_best_cost, list(set(global_best_solution)), global_best_solution, end - start)

# Main Function to run the heuristic based on the parameters passed
def run_meta_heuristics(parameters):
    for param in parameters:
        name = param.get("name")
        hub_numbers = param.get("hub_numbers")
        alphas = param.get("alphas")
        nodes_flow = param.get("data")[0]
        nodes_cost = param.get("data")[1]
        node = param.get("nodes")
        n_iter = param.get("n_iter")
        print("Running RVNS for 10 random initial solutions")
        print("Configuration:",name)

        for n_hub in hub_numbers:
            for alpha in alphas:
                costs = []
                solution_dict = {}
                for i in range(10):
                    result = particle_swarm_heuristic(node, n_hub, nodes_flow, nodes_cost, alpha, n_iter)
                    solution_dict[result[0]] = result
                    print(result)
                    costs.append(result[0])
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
# {"name": "CAB 10", "hub_numbers": [3], "alphas": [0.2], "data":[CAB_10_nodes_flow, CAB_10_nodes_cost], "nodes":10},
# Uncomment all the parameters to run for larger nodes
# prints result in order --> optimal_normalized_cost, optimal_hubs_selection, optimal_configuration, time_taken
 # decrease the n_iter to 10 for quiker results

parameters = [
    {"name": "CAB 10", "hub_numbers": [3, 5], "alphas": [0.2, 0.8], "data": [CAB_10_nodes_flow, CAB_10_nodes_cost],
     "nodes": 10 , "n_iter":1000},
    # {"name": "CAB 25", "hub_numbers": [3,5], "alphas": [0.2, 0.8], "data":[CAB_25_nodes_flow, CAB_25_nodes_cost], "nodes":25, "n_iter":1000},
    # {"name": "TR 55", "hub_numbers": [3,5], "alphas": [0.2,0.8], "data":[TR_55_nodes_flow, TR_55_nodes_cost], "nodes":55, "n_iter":1000},
    # {"name": "TR 81", "hub_numbers": [5,7], "alphas": [0.2,0.8], "data":[TR_81_nodes_flow, TR_81_nodes_cost], "nodes":81, "n_iter":1000},
    # {"name": "TR 100", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data":[TR_100_nodes_flow, TR_100_nodes_cost], "nodes":100, "n_iter":1000},
    # {"name": "TR 130", "hub_numbers": [7,10], "alphas": [0.2,0.8], "data": [TR_130_nodes_flow, TR_130_nodes_cost],"nodes": 130, "n_iter":1000}
]


run_meta_heuristics(parameters)

