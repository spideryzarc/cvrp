import itertools
from builtins import property

import numpy as np
import random as rd
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

import tsp
from queue import PriorityQueue
from copy import deepcopy


class CVRP:
    """
    Representa uma instância de um problema de roteamento de veículos capacitado
    """
    _graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(range(self.n))
        return self._graph

    _c = None

    @property
    def c(self):
        """ matriz de distâncias"""
        if self._c is None:
            self._c = np.round(np.matrix(data=squareform(pdist(self.coord))))
        return self._c

    def __str__(self):
        return self.info['NAME']

    def __init__(self, path: str):
        """
        :param path: Arquivo no formato cvrp da CVRPLIB
        """
        assert os.path.exists(path), path + ' - arquivo não existe.'
        with open(path, 'r') as f:
            self.info = {}
            for ln in f:
                if ln.strip() == 'NODE_COORD_SECTION':
                    break
                self.info[ln.split(':')[0].strip()] = ln.split(':')[1].strip()
            # print(self.info)
            assert self.info['EDGE_WEIGHT_TYPE'] == 'EUC_2D', 'tipo de distância não suportado: ' + self.info[
                'EDGE_WEIGHT_TYPE']
            self.q = int(self.info['CAPACITY'])
            """Capacidade"""
            self.n = int(self.info['DIMENSION'])
            """Número de pontos"""
            self.k = int(self.info['NAME'].split('-k')[-1])
            """Número mínimo de rotas"""
            self.coord = np.zeros(shape=[self.n, 2], dtype=float)
            """Coordenadas no formato matriz nx2"""
            for i in range(self.n):
                v = f.readline().split()
                self.coord[i][0] = float(v[1])
                self.coord[i][1] = float(v[2])
            # print(self.coord)
            for ln in f:
                if ln.strip() == 'DEMAND_SECTION':
                    break
            self.d = np.zeros(self.n, dtype=int)
            """Demandas"""
            for i in range(self.n):
                v = f.readline().split()
                self.d[i] = int(v[1])
        pass

    def plot(self, routes=None, edges=None, clear_edges=True, stop=True, sleep_time=0.1):
        """
        Exibe a instância graficamente

        :param routes: Solução (lista de listas)
        :param edges: lista de arcos (lista de tuplas (i,j) )
        :param clear_edges: limpar o último plot ou não
        :param stop: Parar a execução ou não
        :param sleep_time: Se stop for Falso, tempo de espera em segundos antes de prosseguir
        """
        if clear_edges:
            self.graph.clear_edges()
        if routes is not None:
            for r in routes:
                for i in range(len(r) - 1):
                    self.graph.add_edge(r[i], r[i + 1])
                self.graph.add_edge(r[-1], r[0])
        if edges is not None:
            for i, j in edges:
                self.graph.add_edge(i, j)
        plt.clf()
        color = ['#74BDCB' for i in range(self.n)]
        color[0] = '#FFA384'
        nx.draw_networkx(self.graph, self.coord, with_labels=True, node_size=120, font_size=8, node_color=color)
        if stop:
            plt.show()
        else:
            plt.draw()
            plt.pause(sleep_time)
        pass

    def route_cost(self, routes):
        """
        Calcula o custo da solução

        :param routes: Solução (lista de listas)
        :return : float custo total
        """
        cost = 0
        for r in routes:
            for i in range(1, len(r)):
                cost += self.c[r[i - 1], r[i]]
            cost += self.c[r[-1], r[0]]
        return cost

    def is_feasible(self, routes):
        """
        Verifica se as restrições do problema foram satisfeitas ou não

        :param routes: Solução (lista de listas)
        :return : bool True se for uma solução viável
        """
        if max([self.d[r].sum() for r in routes]) > self.q:
            print("capacidade violada")
            return False
        count = np.zeros(self.n, dtype=int)
        for r in routes:
            for i in r:
                count[i] += 1
        if max(count[1:]) > 1:
            print("cliente vizitado mais de uma vez")
            return False
        if min(count[1:]) < 1:
            print("cliente não vizitado")
            return False
        return True


class Gurobi_CVRP():
    """
    Classe que encapsula um modelo de programação inteira do CVRP
    para resolução com o solver GUROBI
    """

    def _init_model(self):
        m = self.model
        n = self.cvrp.n
        c = self.cvrp.c
        k = self.cvrp.k
        d = self.cvrp.d
        q = self.cvrp.q
        self.x = x = np.zeros(shape=[n, n], dtype=gp.Var)
        """Variáveis do modelo (x_ij)"""

        # variáveis e função objetivo
        for i in range(n):
            for j in range(i):
                x[i, j] = m.addVar(obj=c[i, j], vtype=GRB.BINARY, name='x_%d_%d' % (i, j))
                x[j, i] = m.addVar(obj=c[j, i], vtype=GRB.BINARY, name='x_%d_%d' % (j, i))

        # restrições

        # sum_{i=0}^n x_{ij} = 1 for 1<=j<n
        m.addConstrs(x[:, j].sum() == 1 for j in range(1, n))
        # sum_{j=0}^n x_{ij} = 1 for 1<=i<n
        m.addConstrs(x[i, :].sum() == 1 for i in range(1, n))
        # sum_{j=1}^n x_{0j} = k
        m.addConstr(x[0, 1:].sum() == k)

        if not self.row_generation:
            # sum_{i \in S, j \notin S} x_{ij} >= r(S) \forall S \subset \{1,\ldots,n\}
            for z in range(2, n):
                for s in itertools.combinations(range(1, n), z):
                    print(s)
                    rS = int(np.ceil(d[list(s)].sum() / q))
                    s = set(s)
                    ns = set(range(n)) - s
                    m.addConstr(sum(x[i, j] for i in s for j in ns) >= rS)

        m.update()
        pass

    def __init__(self, cvrp: CVRP, row_generation=True, plot=False):
        """

        :param cvrp: Instância de um CVRP
        :param row_generation: Se a estratégia de geração de linhas deve ser usada ou não
        :param plot: Se as soluções parciais devem ser exibidas ou não
        """
        self.row_generation = row_generation
        self.plot = plot
        self.cvrp = cvrp
        self.model = gp.Model()
        self._init_model()
        # Gerar o 'modelo por extenso' para depuração
        # self.model.write('model.lp')
        pass

    def run(self):
        """
        Executa o solver
        :return: lista de arcos (i,j) da solução ótima
        """

        def cut(model, where):
            """
            Função usada como callback que adiciona restrições omitidas que tenham sido
            violadas

            :param model: Variável do gurobi referente ao modelo em execução
            :param where: Variável do gurobi referente a etapa do algoritmo em execução
            """
            if where == GRB.Callback.MIPSOL:
                n = self.cvrp.n
                edges = [(i, j) for i in range(n) for j in range(n) if
                         i != j and model.cbGetSolution(self.x[i, j]) > 0.5]

                if self.plot:  # solução viável
                    self.cvrp.plot(edges=edges, clear_edges=True, stop=False)

                # separação
                while len(edges) > 0:
                    i, j = edges.pop(0)
                    route = [i, j]
                    while route[0] != j:
                        for k in range(len(edges)):
                            i, j = edges[k]
                            if i == route[-1]:
                                if j != route[0]:
                                    route.append(j)
                                edges.pop(k)
                                break
                    d = sum(self.cvrp.d[route])

                    if d > self.cvrp.q:
                        s = set(route) - {0}
                        ns = set(range(n)) - s
                        lb = int(np.ceil(self.cvrp.d[route].sum() / self.cvrp.q))
                        model.cbLazy(sum(self.x[i, j] for i in s for j in ns) >= lb)
                    elif 0 not in route:
                        ns = set(range(n)) - set(route)
                        model.cbLazy(sum(self.x[i, j] for i in route for j in ns) >= 1)
            pass

        if self.row_generation:
            self.model.Params.lazyConstraints = 1
            self.model.optimize(cut)
        else:
            self.model.optimize()

        n = self.cvrp.n
        edges = [(i, j) for i in range(n) for j in range(n) if i != j and self.x[i, j].X > 0.5]
        return edges


class Heuristicas():
    """
    Classe com método heurísticos para o CVRP
    """
    _saving = None

    @property
    def saving(self):
        """
        Matriz de valores de 'savings' (c[i, 0] + c[0, j] - c[i, j])
        """
        if self._saving is None:
            c = self.cvrp.c
            n = self.cvrp.n
            s = np.zeros(shape=[n, n])
            for i in range(1, n):
                for j in range(1, i):
                    s[i, j] = c[i, 0] + c[0, j] - c[i, j]
                    s[j, i] = c[j, 0] + c[0, i] - c[j, i]
            self._saving = s
        return self._saving

    def __init__(self, cvrp: CVRP, plot=False):
        """

        :param cvrp: Instância de um CVRP
        :param plot: Se as soluções parciais devem ser exibidas ou não
        """
        self.cvrp = cvrp
        self.plot = plot
        self.tabu_list = None
        pass

    def Clarke_n_Wright(self, routes=None):
        """
        Aplica o algoritmo de Clarke and Wright

        :param routes: Solução (lista de listas), caso seja passada uma solução,
        o algoritmo se ocupa de tentar mesclar as rotas existentes nesta solução.
        :return : tupla (custo, solução)
        """
        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q

        # cria n rotas triviais
        if routes is None:
            routes = [[0, i] for i in range(1, n)]
            load = [d[i] for i in range(1, n)]
        else:
            for i in reversed(range(len(routes))):
                if len(routes[i]) <= 1:
                    del routes[i]
            load = [d[r].sum() for r in routes]

        # calcular os 'savings'
        s = self.saving

        cost = self.cvrp.route_cost(routes)
        # concatenar rotas
        while True:
            argmax = None
            maxval = 0
            for k, rk in enumerate(routes):
                for l, rl in enumerate(routes):
                    if (k != l) and maxval < s[rk[-1], rl[1]] and load[k] + load[l] <= q:
                        # adaptação para o tabu
                        if self.tabu_list is not None:
                            if self._is_tabu(rk + rl[1:], cost - s[rk[-1], rl[1]]):
                                continue
                        argmax = k, l
                        maxval = s[rk[-1], rl[1]]

            if argmax is not None:
                # concatenar
                k, l = argmax
                cost -= s[routes[k][-1], routes[l][1]]
                routes[k] = routes[k] + routes[l][1:]
                load[k] += load[l]
                del routes[l]
                del load[l]
                if self.plot:
                    self.cvrp.plot(routes=routes, clear_edges=True, stop=False)
            else:
                break
        assert self.cvrp.is_feasible(routes)
        assert cost == self.cvrp.route_cost(routes)
        return cost, routes

    def _next_fit(self, order):
        """
         Constrói uma solução viável visitando os clientes na ordem estabelecida pela
         lista 'order'

         :param order: lista ordenada de clientes
         :return : solução (lista de listas)
         """
        d = self.cvrp.d
        q = self.cvrp.q
        route = []
        load = 0
        r = [0]
        for i in order:
            if load + d[i] <= q:
                r.append(i)
                load += d[i]
            else:
                route.append(r)
                r = [0, i]
                load = d[i]
            if self.plot:
                self.cvrp.plot(routes=route + [r], clear_edges=True, stop=False)
        route.append(r)
        return route

    def _next_fit_roll(self, order):
        """
        Constrói soluções viáveis visitando os clientes na ordem estabelecida pela lista
        'order' e todas suas rotações.

        :param order: lista ordenada de clientes
        :return : melhor solução explorada(lista de listas)
         """
        best = np.inf
        best_route = None
        for i in range(len(order)):
            route = self._next_fit(np.roll(order, -i))
            # self.VND(route)
            cost = self.cvrp.route_cost(route)
            if best > cost:
                best = cost
                best_route = route
        return best_route

    def tsp_fit(self):
        """
        Constrói soluções viáveis visitando os clientes na ordem estabelecida por
        uma heurística para o PCV/TSP e todas suas rotações.

        :return : melhor solução explorada (lista de listas)
        """
        c = self.cvrp.c
        cost, order = tsp.heuristica1(c)
        route = self._next_fit_roll(order[1:])
        assert self.cvrp.is_feasible(route)
        return route

    def angular_fit(self):
        """
        Constrói soluções viáveis visitando os clientes na ordem estabelecida pela
        varredura angular em relação ao depósito e todas suas rotações.

        :return : melhor solução explorada (lista de listas)
        """
        n = self.cvrp.n
        c = self.cvrp.c
        # coordenadas deslocada para origem
        coord = self.cvrp.coord - self.cvrp.coord[0]
        order = sorted(range(1, n), key=lambda a: np.arctan2(coord[a][0], coord[a][1]), reverse=True)
        route = self._next_fit_roll(order)
        assert self.cvrp.is_feasible(route)
        return route

    def intra_route(self, route, cost=0):
        chg = False
        for r in route:
            imp = True
            while imp:
                imp = tsp.two_opt(r, self.cvrp.c)
                if not imp:
                    imp = tsp.three_opt(r, self.cvrp.c)
                if not imp:
                    imp = tsp.four_opt(r, self.cvrp.c)
                if imp:
                    chg = True
            if self.plot:
                self.cvrp.plot(routes=route, clear_edges=True, stop=False)
        if chg:
            cost = self.cvrp.route_cost(route)
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def _arg_best_insection(self, route, v):
        c = self.cvrp.c
        n = len(route)
        min_arg = n
        min_val = c[route[-1], v] + c[v, route[0]] - c[route[-1], route[0]]
        for i in range(1, n):
            d = c[route[i - 1], v] + c[v, route[i]] - c[route[i - 1], route[i]]
            if d < min_val:
                min_val = d
                min_arg = i
        return min_arg, min_val

    def replace(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        chg = False
        imp = True
        load = [d[r].sum() for r in route]
        while imp:
            imp = False
            for a, ra in enumerate(route):
                for i, vi in enumerate(ra):
                    if i == 0:
                        continue

                    rem_cost = c[ra[i - 1], ra[(i + 1) % len(ra)]] - c[ra[i - 1], ra[i]] - c[
                        ra[i], ra[(i + 1) % len(ra)]]
                    if rem_cost > -1e-3:
                        continue
                    min_val = np.inf
                    min_arg = None
                    for b, rb in enumerate(route):
                        if load[b] + d[vi] <= q and a != b:
                            insert_pos, add_cost = self._arg_best_insection(rb, vi)
                            if add_cost < min_val and add_cost + rem_cost < -1e-3:
                                # adaptação para o tabu
                                if self.tabu_list is not None:
                                    if self._is_tabu(set(ra) - set([vi]), cost + add_cost + rem_cost) or self._is_tabu(
                                            rb + [vi], cost + add_cost + rem_cost):
                                        continue
                                min_val = add_cost
                                min_arg = b, insert_pos
                                if min_val < 1e-3:
                                    break
                    if min_arg is not None and min_val + rem_cost < -1e-3:
                        del ra[i]
                        load[a] -= d[vi]
                        route[min_arg[0]].insert(min_arg[1], vi)
                        load[min_arg[0]] += d[vi]
                        chg = imp = True
                        cost += min_val + rem_cost
                        if self.plot:
                            self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                        break
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def swap(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        imp = True
        chg = False
        load = [d[r].sum() for r in route]
        while imp:
            imp = False
            for a in range(1, len(route)):
                ra = route[a]
                for i in range(1, len(ra)):
                    vi = ra[i]
                    for b in range(a):
                        rb = route[b]
                        for j in range(1, len(rb)):
                            vj = rb[j]
                            if load[a] + d[vj] - d[vi] <= q and load[b] + d[vi] - d[vj] <= q:
                                delta = c[ra[i - 1], vj] + c[vj, ra[(i + 1) % len(ra)]] - c[ra[i - 1], vi] - \
                                        c[vi, ra[(i + 1) % len(ra)]] + c[rb[j - 1], vi] + c[vi, rb[(j + 1) % len(rb)]] - \
                                        c[rb[j - 1], vj] - c[vj, rb[(j + 1) % len(rb)]]
                                if delta < -1e-3:
                                    ra[i] = vj
                                    rb[j] = vi
                                    # adaptação para o tabu
                                    if self.tabu_list is not None:
                                        if self._is_tabu(ra, cost + delta) or self._is_tabu(rb, cost + delta):
                                            ra[i] = vi
                                            rb[j] = vj
                                            continue

                                    load[a] += d[vj] - d[vi]
                                    load[b] += d[vi] - d[vj]
                                    chg = imp = True
                                    vi, vj = vj, vi
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def two_opt_star(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        imp = True
        chg = False
        while imp:
            imp = False
            for a in range(1, len(route)):
                ra = route[a]
                if len(ra) < 3:
                    continue
                for i in range(1, len(ra)):
                    vi = ra[i]
                    vni = ra[(i + 1) % len(ra)]
                    for b in range(a):
                        rb = route[b]
                        if len(rb) < 3:
                            continue
                        for j in range(1, len(rb)):
                            vj = rb[j]
                            vnj = rb[(j + 1) % len(rb)]
                            delta = c[vj, vni] + c[vi, vnj] - c[vi, vni] - c[vj, vnj]
                            if delta < -1e-3:
                                if sum(d[ra[0:i + 1]]) + sum(d[rb[j + 1:]]) <= q and sum(d[rb[0:j + 1]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    # adaptação para o tabu
                                    if self.tabu_list is not None:
                                        if self._is_tabu(ra[0:i + 1] + rb[j + 1:], cost + delta) or self._is_tabu(
                                                rb[0:j + 1] + ra[i + 1:], cost + delta):
                                            continue
                                    na = ra[0:i + 1] + rb[j + 1:]
                                    nb = rb[0:j + 1] + ra[i + 1:]
                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)
                                    chg = imp = True
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                                    break
                            delta = c[vnj, vni] + c[vi, vj] - c[vi, vni] - c[vj, vnj]
                            if delta < -1e-3:
                                if sum(d[ra[:i + 1]]) + sum(d[rb[:j + 1]]) <= q and sum(d[rb[j + 1:]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    # adaptação para o tabu
                                    if self.tabu_list is not None:
                                        if self._is_tabu(ra[:i + 1] + rb[j:0:-1], cost + delta) or self._is_tabu(
                                                [0] + ra[:i:-1] + rb[j + 1:], cost + delta):
                                            continue
                                    na = ra[:i + 1] + rb[j:0:-1]
                                    nb = [0] + ra[:i:-1] + rb[j + 1:]
                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)
                                    chg = imp = True
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                                    break

                        if imp:
                            break
                    if imp:
                        break
                if imp:
                    break

        assert self.cvrp.is_feasible(route)
        return chg, cost

    def two_opt_star_best_imp(self, route, return_at_first=False, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        imp = True
        chg = False
        load = [d[r].sum() for r in route]
        while imp:
            imp = False
            min_delta = 0
            arg = None
            for a in range(1, len(route)):
                ra = route[a]
                for i in range(1, len(ra)):
                    vi = ra[i]
                    vip = ra[(i + 1) % len(ra)]
                    for b in range(a):
                        rb = route[b]
                        for j in range(1, len(rb)):
                            vj = rb[j]
                            vjp = rb[(j + 1) % len(rb)]
                            delta = c[vj, vip] + c[vi, vjp] - c[vi, vip] - c[vj, vjp]
                            if delta < -1e-3:
                                if sum(d[ra[0:i + 1]]) + sum(d[rb[j + 1:]]) <= q and sum(d[rb[0:j + 1]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    if delta < min_delta:
                                        # adaptação para o tabu
                                        if self.tabu_list is not None:
                                            if self._is_tabu(ra[0:i + 1] + rb[j + 1:], cost + delta) or self._is_tabu(
                                                    rb[0:j + 1] + ra[i + 1:], cost + delta):
                                                continue
                                        min_delta = delta
                                        arg = a, ra, b, rb, i, j
            if arg is not None:
                a, ra, b, rb, i, j = arg
                na = ra[0:i + 1] + rb[j + 1:]
                nb = rb[0:j + 1] + ra[i + 1:]
                ra.clear()
                ra.extend(na)
                rb.clear()
                rb.extend(nb)
                load[a] = sum(d[ra])
                load[b] = sum(d[rb])
                chg = imp = True
                cost += min_delta
                if self.plot:
                    self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                if return_at_first:
                    break
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def VND(self, route, cost=None):
        if cost is None:
            cost = self.cvrp.route_cost(route)
        imp = True
        while imp:
            np.random.shuffle(route)
            imp = False
            if not imp:
                imp, cost = self.swap(route, cost)
            if not imp:
                imp, cost = self.replace(route, cost)
            if not imp:
                imp, cost = self.two_opt_star(route, cost)
            if not imp:
                imp, cost = self.intra_route(route, cost)
        # eliminar rotas vazias
        for r in range(len(route) - 1, -1, -1):
            if len(route[r]) < 2:
                del route[r]

        assert self.cvrp.is_feasible(route)
        assert cost == self.cvrp.route_cost(route)
        return cost, route

    def RMS(self, ite: int):
        n = self.cvrp.n
        order = list(range(1, n))
        best = np.inf
        best_route = None
        for i in range(ite):
            np.random.shuffle(order)
            route = self._next_fit_roll(order)
            cost, route = self.VND(route)
            if best > cost:
                best = cost
                best_route = route
                print(i + 1, 'RMS', best)
        return best_route

    def _greedy_random(self, nc: int):
        # Algoritmo de saving
        n = self.cvrp.n
        c = self.cvrp.c
        d = self.cvrp.d
        q = self.cvrp.q

        # cria n rotas triviais
        route = [[0, i] for i in range(1, n)]
        load = [d[i] for i in range(1, n)]

        lista = PriorityQueue()

        # calcular os 'savings'
        s = self.saving

        # concatenar rotas
        while True:
            lista.queue.clear()
            for k, rk in enumerate(route):
                for l, rl in enumerate(route):
                    if (k != l) and load[k] + load[l] <= q:
                        saving = s[rk[-1], rl[1]]
                        if lista.qsize() < nc or saving > lista.queue[0][0]:
                            lista.put((saving, k, l))
                            if lista.qsize() > nc:
                                lista.get()
            if lista.qsize() > 0:
                # concatenar
                i = np.random.randint(0, lista.qsize())
                # w = np.array([x[0] for x in lista.queue])**-1
                # i = rd.choices(range(lista.qsize()), weights=w)[0]
                x, k, l = lista.queue[i]
                route[k] = route[k] + route[l][1:]
                # tsp.two_opt(route[k],c)
                load[k] += load[l]
                del route[l]
                del load[l]
                if self.plot:
                    self.cvrp.plot(routes=route, clear_edges=True, stop=False)
            else:
                break
        assert self.cvrp.is_feasible(route)
        return route

    def GRASP(self, ite: int, nc: int):
        n = self.cvrp.n
        order = list(range(1, n))
        best_cost = np.inf
        best_route = None
        for i in range(ite):
            np.random.shuffle(order)
            route = self._greedy_random(nc)
            self.VND(route)
            cost = self.cvrp.route_cost(route)
            if best_cost > cost:
                best_cost = cost
                best_route = route
                print(i + 1, 'GRASP', best_cost)
        return best_cost, best_route

    def _shake(self, route, k=1, tenure=0):
        # seleciona k rotas para a destruição
        destruct_list = list(range(len(route)))
        np.random.shuffle(destruct_list)
        destruct_list = sorted(destruct_list[:k], reverse=True)

        # destroi rotas
        v = []
        cost = self.cvrp.route_cost(route)
        for r in destruct_list:
            v.extend(route[r][1:])
            if self.tabu_list is not None:
                # cria tabu
                self.tabu_list.append((set(route[r]), cost))
                if len(self.tabu_list) > tenure:
                    del self.tabu_list[0]
            del route[r]

        # cria n rotas triviais
        for r in v:
            route.append([0, r])

        cost, route = self.Clarke_n_Wright(route)
        return cost, route

    def _is_tabu(self, r, cost):
        if r is not set:
            r = set(r)
        for s, c in self.tabu_list:
            if s == r and c <= cost:
                return True
        return False

    def _has_tabu(self, route):
        for r in route:
            if set(r) in self.tabu_list:
                return True
        return False

    def tabu_search(self, ite: int, k: int, tenure: int, reset_factor=1.05):
        self.tabu_list = []
        n = self.cvrp.n
        best_cost, best_route = self.Clarke_n_Wright()
        best_cost, best_route = self.VND(best_route, best_cost)
        print(0, 'Tabu', best_cost)
        assert best_cost == self.cvrp.route_cost(best_route)
        route = deepcopy(best_route)
        for i in range(ite):
            cost, route = self._shake(route, k, tenure)
            cost, route = self.VND(route, cost)
            assert cost == self.cvrp.route_cost(route)
            # print(cost)
            if best_cost > cost:
                self.tabu_list.clear()
                best_cost = cost
                best_route = deepcopy(route)
                print(i + 1, 'Tabu', best_cost)
                if self.plot:
                    self.cvrp.plot(routes=route, clear_edges=True, stop=False)
            elif best_cost * reset_factor < cost:
                self.tabu_list.clear()
                route = deepcopy(best_route)
                # print('reset', cost)

        self.tabu_list = None
        return best_cost, best_route

    def _get_edges_set(self, route):
        edges = set()
        for r in route:
            edges.add((0, r[-1]))
            for i in range(len(r)):
                edges.add(tuple(sorted((r[i - 1], r[i]))))
        return edges

    def _route_dist(self, a, b):
        edges_a = self._get_edges_set(a)
        edges_b = self._get_edges_set(b)
        return len(edges_a - edges_b)

    def _ref_set_update(self, pop, size, elite=1):
        pop.sort()

        # remover individuos repetidos
        for i in range(len(pop) - 1, -1, -1):
            for j in range(i):
                if self._route_dist(pop[i][1], pop[j][1]) == 0:
                    del pop[i]
                    break

        pop = pop[:max(int(elite * len(pop)), size)]
        n = len(pop)
        d = np.zeros(shape=[n, n])
        for i in range(n):
            for j in range(i):
                d[i, j] = d[j, i] = self._route_dist(pop[i][1], pop[j][1])
        p = pop[0]
        ref = [p]
        dist = np.zeros(n)
        for i in range(n):
            dist[i] = d[0, i]
        # print(max(dist))
        for k in range(1, size):
            p_idx = np.argmax(dist)
            p = pop[p_idx]
            ref.append(p)
            for i in range(n):
                dist[i] = min(dist[i], d[p_idx, i])
        return ref

    def _comb_sols(self, sols):
        n = self.cvrp.n
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        petalas = []
        for c, s in sols:
            for r in s:
                petalas.append(r)
        np.random.shuffle(petalas)
        route = []
        for p in petalas:
            r = [0] + [v for v in p if not visited[v]]
            if len(r) >= 2:
                visited[r] = True
                route.append(r)
        assert self.cvrp.is_feasible(route)
        cost, route = self.Clarke_n_Wright(route)
        cost, route = self.VND(route, cost)
        # cost, route = self.VND(route)
        return cost, route

    def scatter_search(self, ite=100, ini_pop_size=20, ref_size=10, subset_size=3, grasp_k=10, diver=.7,
                       mut_factor=0.1):
        pop = []
        print('Inicializando população')
        order = list(range(1, self.cvrp.n))
        for i in range(ini_pop_size):
            route = self._greedy_random(grasp_k)
            cost, route = self.VND(route)
            print(i + 1, 'Pop.', cost)
            pop.append((cost, route))
        ref_set = self._ref_set_update(pop, ref_size, diver)
        best_cost, best_route = ref_set[0]
        print('0 SS', best_cost)
        if self.plot:
            self.cvrp.plot(routes=best_route, clear_edges=True, stop=False)
        for i in range(ite):
            print('Geração', i + 1)
            offspring = []
            for sub_set in itertools.combinations(ref_set, subset_size):
                s = self._comb_sols(sub_set)
                offspring.append(s)

            for k in range(len(offspring)):
                if rd.random() < mut_factor:
                    # print('Mutação')
                    cost, route = self._shake(offspring[k][1], k=2)
                    cost, route = self.VND(route)
                    offspring[k] = cost, route
            ref_set = self._ref_set_update(ref_set + offspring, ref_size, diver)
            cost, route = ref_set[0]
            if cost < best_cost:
                best_cost, best_route = deepcopy(ref_set[0])
                print(i + 1, 'SS', best_cost)
                if self.plot:
                    self.cvrp.plot(routes=best_route, clear_edges=True, stop=False)

        return best_cost, best_route
