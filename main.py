from cvrp import CVRP, Gurobi_CVRP, Heuristicas
import numpy as np

if __name__ == '__main__':
    if __debug__:
        print('modo debug ATIVADO')
    else:
        print('mode debud DESATIVADO')
    np.random.seed(7)

    # cvrp = CVRP('instances/toy-n11-k.vrp.txt')
    # cvrp = CVRP('instances/A-n32-k5.vrp.txt')
    cvrp = CVRP('instances/A-n80-k10.vrp.txt')
    # model = Gurobi_CVRP(cvrp,row_generation=True, plot=False)
    # edges = model.run()
    # cvrp.plot(edges=edges)
    # exit(0)

    # cvrp = CVRP('instances/X-n1001-k43.vrp.txt')

    print(cvrp)

    heuristicas = Heuristicas(cvrp, plot=False)
    # route = heuristicas.Clarke_n_Wright()
    # route = heuristicas.tsp_fit()
    # route = heuristicas.angular_fit()

    # route = heuristicas.RMS(100)
    # heuristicas.plot = True
    # heuristicas.VND(route)
    # route = heuristicas.GRASP(1000, 3)
    route = heuristicas.tabu_search(1000, 2, 20)
    # print(cvrp.route_cost(route))
    # cvrp.plot(route=route)

    print(cvrp.route_cost(route))
    cvrp.plot(route=route)
