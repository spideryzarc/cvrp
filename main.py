from cvrp import CVRP, Gurobi_CVRP, Heuristicas
import numpy as np
import random as rd
from cProfile import run as profile

if __name__ == '__main__':
    if __debug__:
        print('modo debug ATIVADO')
    else:
        print('mode debug DESATIVADO')
    np.random.seed(7)
    rd.seed(7)

    # cvrp = CVRP('instances/toy-n11-k.vrp.txt')
    # cvrp = CVRP('instances/A-n32-k5.vrp.txt')
    # cvrp = CVRP('instances/A-n80-k10.vrp.txt')
    cvrp = CVRP('instances/X-n1001-k43.vrp.txt')
    heuristicas = Heuristicas(cvrp, plot=False)

    # model = Gurobi_CVRP(cvrp,row_generation=True, plot=False)
    # cost, route = heuristicas.GRASP(10, 3)
    # print(cost)
    # edges = model.run(route)
    # cvrp.plot(edges=edges)
    # exit(0)

    print(cvrp)

    # cost,route = heuristicas.Clarke_n_Wright()
    profile('cost,route = heuristicas.Clarke_n_Wright()')
    
    # route = heuristicas.tsp_fit()
    # route = heuristicas.angular_fit()

    # route = heuristicas.RMS(100)
    # heuristicas.plot = True
    # heuristicas.VND(route)
    # cost,route = heuristicas.GRASP(100, 3)
    # cost, route = heuristicas.tabu_search(500, 2, 20)
    # print(cvrp.route_cost(route))
    # cvrp.plot(route=route)
    # cost, route = heuristicas.scatter_search(ite=50, ini_pop_size=100, ref_size=5, subset_size=3)
    # cost, route = heuristicas.ils(500, 1)
    # cost, route = heuristicas.ant_colony(ite=1, ants=1000, online=True, elitist=True, evapor=0.3)
    # cost, route = heuristicas.ant_colony(ite=50, ants=20, online=False, update_by='rank', k=5, worst=True,
    #                                      elitist=True, evapor=0.5)
    # cost, route = heuristicas.ant_colony(ite=50, ants=20, online=False, update_by='quality', k=5, worst=True,
    #                                      elitist=True, evapor=0.3)
    print(cvrp.route_cost(route))
    cvrp.plot(routes=route)
