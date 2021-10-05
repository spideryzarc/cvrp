import numpy as np
# Biblioteca para trabalhar com grafos
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random as rd
from assignment import assignment


def randomGraph(n, W=1000, H=1000):
    coordinates = []
    for i in range(n):
        coordinates.append((rd.randint(0, W), rd.randint(0, H)))
    dist_matrix = squareform(pdist(coordinates))
    G = nx.complete_graph(n)
    for i in range(n):
        G.nodes[i]["coord"] = coordinates[i]
        G.nodes[i]["label"] = i
        for j in range(i):
            G.edges[i, j]['weight'] = G.edges[j, i]['weight'] = dist_matrix[i][j]
    return G


# TODO corrigir erro quando tem arestas infinitas validas devido ao branch
def NearestNeighbor(mat):
    # assert isinstance(mat, np.matrix), "use np.matrix"
    n = len(mat)
    used = np.zeros(n, dtype=bool)
    last = 0  # rd.randint(0, n - 1)
    used[last] = True
    cost = 0
    route = [last]
    for i in range(n - 1):
        minarg = -1
        min = np.inf
        for j in range(n):
            if (not used[j] and mat[last, j] < min):
                min = mat[last, j]
                minarg = j
        if minarg >= 0:
            cost += min
            last = minarg
            route.append(last)
            used[last] = True

    cost += mat[last, route[0]]
    return cost, np.roll(route, - np.argmin(route))


def two_opt(route, mat: np.matrix):
    def opt2CostDelta(mat, i, ni, j, nj):
        return mat[i, j] + mat[ni, nj] - mat[i, ni] - mat[j, nj]

    imp = True
    changed = False
    n = len(route)
    while imp:
        imp = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if opt2CostDelta(mat, route[i], route[i + 1], route[j], route[j + 1]) < 0:
                    route[i + 1:j + 1] = route[j:i:-1]
                    changed = imp = True
        for i in range(1, n - 2):
            if opt2CostDelta(mat, route[i], route[i + 1], route[-1], route[0]) < 0:
                route[:] = np.roll(route, -1)
                route[i: -1] = route[n - 2:i - 1:-1]
                route[:] = np.roll(route, 1)
                changed = imp = True
    return changed


def three_opt(route, mat: np.matrix):
    imp = True
    changed = False
    n = len(route)
    while imp:
        imp = False
        for i in range(n - 2):
            ri = route[i]
            rni = route[i + 1]
            for j in range(i + 2, n - 1):
                rj = route[j]
                rnj = route[j + 1]
                for k in range(j + 2, n - 1):
                    rk = route[k]
                    rnk = route[k + 1]
                    rem = mat[ri, rni] + mat[rj, rnj] + mat[rk, rnk]
                    # 0 -- i -> ni -- j -> nj -- k -> nk -- 0
                    # 0 -- i -> nj -- k -> ni -- j -> nk -- 0
                    if mat[ri, rnj] + mat[rk, rni] + mat[rj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j + 1:k + 1]) + list(route[i + 1:j + 1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        # print('3opt 1')
                        break
                    # 0 -- i -> ni -- j -> nj -- k -> nk -- 0
                    # 0 -- i -> nj -- k -> (j -- ni) -> nk -- 0
                    if mat[ri, rnj] + mat[rk, rj] + mat[rni, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j + 1:k + 1]) + list(route[j:i:-1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        # print('3opt 2')
                        break

                    # 0 -- i -> ni -- j -> nj -- k -> nk -- 0
                    # 0 -- i -> (k -- nj) -> ni -- j -> nk -- 0
                    if mat[ri, rk] + mat[rnj, rni] + mat[rj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[k:j:-1]) + list(route[i + 1:j + 1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        # print('3opt 3')
                        break

                    # 0 -- i -> ni -- j -> nj -- k -> nk -- 0
                    # 0 -- i -> (j -- ni) -> (k -- nj) -> nk -- 0
                    if mat[ri, rj] + mat[rni, rk] + mat[rnj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j:i:-1]) + list(route[k:j:-1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        # print('3opt 4')
                        break
                if imp:
                    break
    return changed


def four_opt(route, mat: np.matrix):
    #cross
    imp = True
    changed = False
    n = len(route)
    while imp:
        imp = False
        for i in range(n - 2):
            ri = route[i]
            rni = route[i + 1]
            for j in range(i + 2, n - 1):
                rj = route[j]
                rnj = route[j + 1]
                for k in range(j + 2, n - 1):
                    rk = route[k]
                    rnk = route[k + 1]
                    for l in range(k + 2, n - 1):
                        rl = route[l]
                        rnl = route[l + 1]
                        rem = mat[ri, rni] + mat[rj, rnj] + mat[rk, rnk] + mat[rl, rnl]
                        # 0 -- i -> ni -- j -> nj -- k -> nk -- l -> nl -- 0
                        # 0 -- i -> nk -- l -> nj -- k -> ni -- j -> nl -- 0
                        if mat[ri, rnk] + mat[rl, rnj] + mat[rk, rni] + mat[rj, rnl] < rem:
                            r = list(route[0:i + 1]) + list(route[k + 1:l + 1]) + list(route[j + 1:k + 1]) \
                                + list(route[i + 1:j + 1]) + list(route[l + 1:])
                            route[:] = r
                            changed = imp = True
                            print('4opt 1')
                            return True
    return changed


def cost(route, mat: np.matrix):
    c = mat[route[0], route[-1]]
    for i in range(1, len(route)):
        c += mat[route[i - 1], route[i]]
    return c


def _best_insection(route, v, mat: np.matrix):
    n = len(route)
    minArg = n
    min = mat[route[n - 1], v] + mat[v, route[0]] - mat[route[n - 1], route[0]]
    for i in range(1, n):
        d = mat[route[i - 1], v] + mat[v, route[i]] - mat[route[i - 1], route[i]]
        if d < min:
            min = d
            minArg = i
    route.insert(minArg, v)
    return


def further_insection(mat: np.matrix):
    route = [0]
    n = len(mat)
    arg = None
    max = 0
    for i in range(n):
        for j in range(i):
            if max < mat[i, j]:
                max = mat[i, j]
                arg = (i, j)
    route = list(arg)
    dist = np.zeros(n)
    dist[route[0]] = dist[route[1]] = -np.inf
    for i in range(n):
        if dist[i] != -np.inf:
            dist[i] = min(mat[route[0], i], mat[route[1], i])
    for i in range(n - 2):
        p = dist.argmax()
        _best_insection(route, p, mat)
        dist[p] = -np.inf
        for j in range(n):
            if dist[j] != -np.inf:
                dist[i] = min(dist[i], mat[p, j])
    r = np.roll(route, - np.argmin(route))
    return r


def heuristica1(mat):
    n = len(mat)
    matb = np.zeros([n, n])
    dcost, x, v, u = assignment(mat)
    pi = np.zeros(n)
    for i in range(n):
        pi[i] = -0.5 * (v[i] + u[i])
    for i in range(n):
        for j in range(i):
            matb[i, j] = matb[j, i] = np.round(mat[i, j] + pi[i] + pi[j], 3)
    ub, route = NearestNeighbor(matb)
    while two_opt(route, mat) or three_opt(route, mat) or four_opt(route, mat):
        pass
    return cost(route, mat), route


def route_to_graph(route, graph):
    graph.clear_edges()
    for i in range(0, len(route)):
        if route[i - 1] < route[i]:
            graph.add_edge(route[i - 1], route[i])
        else:
            graph.add_edge(route[i], route[i - 1])
    return


def graph_to_route(graph):
    n = len(graph.edges)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    route = [0]
    p = 0
    added = True
    while added:
        added = False
        for i in graph.neighbors(p):
            if not visited[i]:
                route.append(i)
                p = i
                visited[i] = True
                added = True
                break

    return route
