import itertools
from builtins import property, reversed

import numpy as np
import random as rd
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from collections import namedtuple

from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

import tsp
from queue import PriorityQueue
from copy import deepcopy

import time


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return timed


def progress(done, total, text: str):
    x = int(round(40.0 * done / total))
    print(f"\r{text}: |{'█' * x}{'-' * (40 - x)}|", end='')
    if done == total:
        print()
    pass


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

    def plot(self, routes=None, edges=None, clear_edges=True, stop=True, sleep_time=0.01):
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
                if len(r) > 1:
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

    def run(self, route=None):
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

        if route is not None:
            # initial solution
            for r in route:
                for i in range(1, len(r)):
                    self.x[r[i - 1], r[i]].Start = 1
                self.x[r[-1], r[0]].Start = 1

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

    _max_saving = None

    @property
    def max_saving(self):
        if self._max_saving is None:
            s = self.saving
            self._max_saving = [s[i, :].max() for i in range(len(s))]
        return self._max_saving

    # @timeit
    def Clarke_n_Wright(self, routes=None):
        """
        Aplica o algoritmo de Clarke and Wright paralelo

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
        else:
            for i in reversed(range(len(routes))):
                if len(routes[i]) <= 1:
                    del routes[i]

        load_r_zipped = [[d[r].sum(), r] for r in routes]
        # calcular os 'savings'
        s = self.saving

        cost = self.cvrp.route_cost(routes)
        # concatenar rotas
        max_s = self.max_saving
        while True:
            argmax = None
            max_val = 0
            load_r_zipped.sort(key=lambda a: max_s[a[1][-1]], reverse=True)
            for k, rk in enumerate(load_r_zipped):
                if max_s[rk[1][-1]] <= max_val:
                    break
                for l, rl in enumerate(load_r_zipped):
                    if (k != l) and max_val < s[rk[1][-1], rl[1][1]] and rk[0] + rl[0] <= q:
                        # adaptação para o tabu
                        if self.tabu_list is not None:
                            if self._is_tabu(rk[1] + rl[1][1:], cost - s[rk[1][-1], rl[1][1]]):
                                continue
                        argmax = k, l
                        max_val = s[rk[1][-1], rl[1][1]]

            if argmax is not None:
                # concatenar
                k, l = argmax
                cost -= s[load_r_zipped[k][1][-1], load_r_zipped[l][1][1]]
                load_r_zipped[k][1].extend(load_r_zipped[l][1][1:])
                load_r_zipped[l][1].clear()
                load_r_zipped[k][0] += load_r_zipped[l][0]
                del load_r_zipped[l]
                if self.plot:
                    self.cvrp.plot(routes=routes, clear_edges=True, stop=False)
            else:
                break

        # remover rotas vazias
        for i in reversed(range(len(routes))):
            if len(routes[i]) <= 1:
                del routes[i]

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
                # if not imp:
                #     imp = tsp.four_opt(r, self.cvrp.c)
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

    # def two_opt_star_best_imp(self, route, return_at_first=False, cost=0):
    #     q = self.cvrp.q
    #     c = self.cvrp.c
    #     d = self.cvrp.d
    #     imp = True
    #     chg = False
    #     load = [d[r].sum() for r in route]
    #     while imp:
    #         imp = False
    #         min_delta = 0
    #         arg = None
    #         for a in range(1, len(route)):
    #             ra = route[a]
    #             for i in range(1, len(ra)):
    #                 vi = ra[i]
    #                 vip = ra[(i + 1) % len(ra)]
    #                 for b in range(a):
    #                     rb = route[b]
    #                     for j in range(1, len(rb)):
    #                         vj = rb[j]
    #                         vjp = rb[(j + 1) % len(rb)]
    #                         delta = c[vj, vip] + c[vi, vjp] - c[vi, vip] - c[vj, vjp]
    #                         if delta < -1e-3:
    #                             if sum(d[ra[0:i + 1]]) + sum(d[rb[j + 1:]]) <= q and sum(d[rb[0:j + 1]]) + sum(
    #                                     d[ra[i + 1:]]) <= q:
    #                                 if delta < min_delta:
    #                                     # adaptação para o tabu
    #                                     if self.tabu_list is not None:
    #                                         if self._is_tabu(ra[0:i + 1] + rb[j + 1:], cost + delta) or self._is_tabu(
    #                                                 rb[0:j + 1] + ra[i + 1:], cost + delta):
    #                                             continue
    #                                     min_delta = delta
    #                                     arg = a, ra, b, rb, i, j
    #         if arg is not None:
    #             a, ra, b, rb, i, j = arg
    #             na = ra[0:i + 1] + rb[j + 1:]
    #             nb = rb[0:j + 1] + ra[i + 1:]
    #             ra.clear()
    #             ra.extend(na)
    #             rb.clear()
    #             rb.extend(nb)
    #             load[a] = sum(d[ra])
    #             load[b] = sum(d[rb])
    #             chg = imp = True
    #             cost += min_delta
    #             if self.plot:
    #                 self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
    #             if return_at_first:
    #                 break
    #     assert self.cvrp.is_feasible(route)
    #     return chg, cost

    def VND(self, sol, cost=None):
        """
        Variable Neighborhood Descent
        :param sol: Solução (lista de listas)
        :param cost: Custo atual da solução
        :return: tupla (custo, solução)
        """
        if cost is None:
            cost = self.cvrp.route_cost(sol)
        imp = True
        while imp:
            np.random.shuffle(sol)
            imp = False
            if not imp:
                imp, cost = self.swap(sol, cost)
            if not imp:
                imp, cost = self.replace(sol, cost)
            if not imp:
                imp, cost = self.two_opt_star(sol, cost)
            if not imp:
                imp, cost = self.intra_route(sol, cost)

        # eliminar rotas vazias
        for i in reversed(range(len(sol))):
            if len(sol[i]) <= 1:
                del sol[i]

        assert self.cvrp.is_feasible(sol)
        assert cost == self.cvrp.route_cost(sol)
        return cost, sol

    def RMS(self, ite: int):
        """
        Random Multistart

        :param ite: número de iterações
        :return: tupla (custo, solução)
        """
        n = self.cvrp.n
        order = list(range(1, n))
        best_cost = np.inf
        best_sol = None
        for i in range(ite):
            np.random.shuffle(order)
            sol = self._next_fit(order)
            cost, sol = self.VND(sol)
            if best_cost > cost:
                best_cost = cost
                best_sol = sol
                print(i + 1, 'RMS', best_cost)
        return best_cost, best_sol

    def _greedy_random(self, nc: int):
        # Algoritmo de saving
        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q

        # cria n rotas triviais
        routes = [[0, i] for i in range(1, n)]
        # load = [d[i] for i in range(1, n)]
        load_r_zipped = [[d[r].sum(), r] for r in routes]

        lista = PriorityQueue()

        # calcular os 'savings'
        s = self.saving
        cost = self.cvrp.route_cost(routes)
        max_s = self.max_saving
        # concatenar rotas
        while True:
            lista.queue.clear()
            max_val = 0
            load_r_zipped.sort(key=lambda a: max_s[a[1][-1]], reverse=True)
            for k, rk in enumerate(load_r_zipped):
                if max_s[rk[1][-1]] <= max_val:
                    break
                for l, rl in enumerate(load_r_zipped):
                    if (k != l) and max_val < s[rk[1][-1], rl[1][1]] and rk[0] + rl[0] <= q:
                        saving = s[rk[1][-1], rl[1][1]]
                        if lista.qsize() < nc or saving > lista.queue[0][0]:
                            lista.put((saving, k, l))
                            if lista.qsize() > nc:
                                lista.get()
                            max_val = lista.queue[0][0]
            if lista.qsize() > 0:
                # concatenar
                i = np.random.randint(0, lista.qsize())
                x, k, l = lista.queue[i]
                cost -= s[load_r_zipped[k][1][-1], load_r_zipped[l][1][1]]
                load_r_zipped[k][1].extend(load_r_zipped[l][1][1:])
                load_r_zipped[l][1].clear()
                load_r_zipped[k][0] += load_r_zipped[l][0]
                del load_r_zipped[l]
                if self.plot:
                    self.cvrp.plot(routes=routes, clear_edges=True, stop=False)
            else:
                break

        for i in reversed(range(len(routes))):
            if len(routes[i]) <= 1:
                del routes[i]
        assert self.cvrp.is_feasible(routes)
        assert cost == self.cvrp.route_cost(routes)
        return cost, routes

    def GRASP(self, ite: int, nc: int):
        n = self.cvrp.n
        order = list(range(1, n))
        best_cost = np.inf
        best_route = None
        for i in range(ite):
            np.random.shuffle(order)
            cost, route = self._greedy_random(nc)
            cost, route = self.VND(route, cost)
            if best_cost > cost:
                best_cost = cost
                best_route = route
                print(i + 1, 'GRASP', best_cost)
        return best_cost, best_route

    def _shake(self, sol, cost, k=1, tenure=0):
        """
        Perturba a solução destruindo rotas e reconstruindo com algoritmo de saving
        As rotas destruídas se tornam tabus e não poderão ser reconstruidas

        :param sol: Solução (lista de listas)
        :param k: número de rotas a serem destruídas
        :param tenure: tamanho máximo da lista tabu, usado quando self.tabu_list não é None
        :return: tupla (custo, solução)
        """
        # seleciona k rotas para a destruição
        destruct_list = sorted(rd.sample(range(len(sol)), k), reverse=True)

        v = []
        # clientes sem rota
        for r in destruct_list:
            # destruir rotas
            v.extend(sol[r][1:])
            if self.tabu_list is not None:
                # cria tabu
                self.tabu_list.append((set(sol[r]), cost))
                if len(self.tabu_list) > tenure:
                    del self.tabu_list[0]
            del sol[r]

        # cria rotas triviais para os clientes sem rotas
        for i in v:
            sol.append([0, i])
        cost = self.cvrp.route_cost(sol)
        return cost, sol

    def _is_tabu(self, r: [list, set], cost: float):
        """
        Verifica se uma rota é ou não um tabu
        :param r: rota de um veículo
        :param cost: critério de aspiração, se o custo for bom, o tabu será ignorado
        :return: bool se r é ou não um tabu
        """
        if r is not set:
            r = set(r)
        for s, c in self.tabu_list:
            if s == r and c <= cost:
                return True
        return False

    @timeit
    def tabu_search(self, ite: int, k: int, tenure: int, reset_factor=1.05):
        """
        Executa a meta-heurística tabu search

        :param ite: número de iterações
        :param k: número de rotas destruídas por pertubação
        :param tenure: quantidade máxima de regras tabu mantidas
        :param reset_factor: desvio máximo entre a melhor solução e a solução corrente para forçar reset da lista tabu
        :return: tupla (custo, solução)
        """
        self.tabu_list = []
        best_cost, best_sol = self.Clarke_n_Wright()
        best_cost, best_sol = self.VND(best_sol, best_cost)
        print(0, 'Tabu', best_cost)
        current_sol = deepcopy(best_sol)
        current_cost = best_cost
        for i in range(ite):
            current_cost, current_sol = self._shake(current_sol, current_cost, k, tenure)
            current_cost, current_sol = self.VND(current_sol, current_cost)
            if best_cost > current_cost:
                self.tabu_list.clear()
                best_cost = current_cost
                best_sol = deepcopy(current_sol)
                print(i + 1, 'Tabu', best_cost)
                if self.plot:
                    self.cvrp.plot(routes=current_sol, clear_edges=True, stop=False)
            elif best_cost * reset_factor < current_cost:
                self.tabu_list.clear()
                current_sol = deepcopy(best_sol)

        self.tabu_list = None
        return best_cost, best_sol

    @timeit
    def ils(self, ite: int, k: int, reset_factor=1.05):
        """
        Executa a meta-heurística iterated local search

        :param ite: número de iterações
        :param k: número de rotas destruídas por pertubação
        :param reset_factor: desvio máximo entre a melhor solução e a solução corrente para forçar reset da lista tabu
        :return: tupla (custo, solução)
        """
        best_cost, best_sol = self.Clarke_n_Wright()
        best_cost, best_sol = self.VND(best_sol, best_cost)
        print(0, 'ILS', best_cost)
        current_sol = deepcopy(best_sol)
        current_cost = best_cost
        for i in range(ite):
            current_cost, current_sol = self._shake(current_sol, current_cost, k)
            current_cost, current_sol = self.VND(current_sol, current_cost)
            if self.plot:
                self.cvrp.plot(routes=current_sol, clear_edges=True, stop=False)
            if best_cost > current_cost:
                best_cost = current_cost
                best_sol = deepcopy(current_sol)
                print(i + 1, 'ILS', best_cost)
                if self.plot:
                    self.cvrp.plot(routes=current_sol, clear_edges=True, stop=False)
            elif best_cost * reset_factor < current_cost:
                # print('reset')
                current_sol = deepcopy(best_sol)
        return best_cost, best_sol

    @staticmethod
    def _get_edges_set(route):
        edges = set()
        for r in route:
            edges.add((0, r[-1]))
            for i in range(len(r)):
                edges.add(tuple(sorted([r[i - 1], r[i]])))
        return edges

    # def _route_dist(self, a, b):
    #     edges_a = self._get_edges_set(a)
    #     edges_b = self._get_edges_set(b)
    #     return len(edges_a - edges_b)

    def _ref_set_update(self, pop, ref_size, diver=1):
        pop.sort()
        # remover individuos repetidos
        pop_size = len(pop)
        edges = [self._get_edges_set(pop[i][1]) for i in range(pop_size)]
        for i in reversed(range(len(pop))):
            for j in range(i):
                if edges[i] == edges[j]:
                    del pop[i]
                    del edges[i]
                    break
        pop = pop[:max(int(diver * pop_size), ref_size)]
        pop_size = len(pop)
        d = np.zeros(shape=[pop_size, pop_size])
        for i in range(pop_size):
            for j in range(i):
                d[i, j] = d[j, i] = len(edges[i] - edges[j])

        p = pop[0]
        ref = [p]
        dist = np.zeros(pop_size)
        for i in range(pop_size):
            dist[i] = d[0, i]
        # print(max(dist))
        for k in range(1, ref_size):
            p_idx = np.argmax(dist)
            p = pop[p_idx]
            ref.append(p)
            for i in range(pop_size):
                dist[i] = min(dist[i], d[p_idx, i])

        return ref

    # def _comb_sols(self, sols: list):
    #     """
    #     Combina uma lista de soluções em uma nova solução
    #     :param sols: lista de soluções
    #     :return: tupla (custo, solução)
    #     """
    #     n = self.cvrp.n
    #     d = self.cvrp.d
    #     c = self.cvrp.c
    #     visited = np.zeros(n, dtype=bool)
    #     visited[0] = True
    #     petalas = []
    #     for cost, sol in sols:
    #         for r in sol:
    #             petalas.append((tsp.cost(r, c), deepcopy(r)))
    #
    #     new_sol = []
    #
    #     while len(petalas) > 0:
    #         p = max(petalas, key=lambda a: d[a[1]].sum() / a[0])
    #
    #         if len(p[1]) >= 2:
    #             visited[p[1]] = True
    #             new_sol.append(p[1])
    #
    #         petalas.remove(p)
    #
    #         for score, r in petalas:
    #             for i in reversed(range(1, len(r))):
    #                 if visited[r[i]]:
    #                     del r[i]
    #         for i in reversed(range(len(petalas))):
    #             if len(petalas[i][1]) < 2:
    #                 del petalas[i]
    #
    #     assert self.cvrp.is_feasible(new_sol)
    #     cost, new_sol = self.Clarke_n_Wright(new_sol)
    #     # if len(new_sol) == len(sols[1][1]):
    #     cost, new_sol = self.VND(new_sol, cost)
    #
    #     return cost, new_sol

    def _recombination(self, sols: list):
        """
        Combina uma lista de soluções em uma nova solução
        :param sols: lista de soluções
        :return: tupla (custo, solução)
        """
        n = self.cvrp.n
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        petalas = []
        for c, s in sols:
            for r in s:
                petalas.append(r)
        np.random.shuffle(petalas)
        new_sol = []
        for p in petalas:
            r = [0] + [v for v in p if not visited[v]]
            if len(r) >= 2:
                visited[r] = True
                new_sol.append(r)

        assert self.cvrp.is_feasible(new_sol)
        cost, new_sol = self.Clarke_n_Wright(new_sol)
        cost, new_sol = self.VND(new_sol, cost)

        return cost, new_sol

    @timeit
    def scatter_search(self, ite=100, ini_pop_size=20, ref_size=10, subset_size=3, grasp_k=5, diver=.7,
                       mut_factor=0.02):
        """
        Aplica a meta-heurística de busca difusa

        :param ite: número de iterações ou gerações
        :param ini_pop_size: tamanho da população inicial
        :param ref_size: tamanho do conjunto de referência
        :param subset_size: número de indivíduos combinados por vez
        :param grasp_k: número de candidatos usados no grasp de geração de soluções iniciais
        :param diver: [0 1] percentual dos melhores indivíduos a serem mantidos antes do critério de dispersão ser aplicado
        :param mut_factor: probabilidade de uma solução gerada sofre mutação
        :return: tupla (custo, solução)
        """
        pop = []
        # print('Inicializando população')
        for i in range(ini_pop_size):
            cost, route = self._greedy_random(grasp_k)
            cost, route = self.VND(route, cost)
            progress(i + 1, ini_pop_size, 'Inicializando população')
            pop.append((cost, route))

        print('\nSelecionando conjunto de referência')
        ref_set = self._ref_set_update(pop, ref_size, diver)
        # for i, (cost, route) in enumerate(ref_set):
        #     ref_set[i] = self.VND(route, cost)
        # ref_set.sort()
        best_cost, best_route = ref_set[0]
        print('0 SS', best_cost)
        if self.plot:
            self.cvrp.plot(routes=best_route, clear_edges=True, stop=False)
        for i in range(ite):
            print(f'\rIteração  {i + 1}   ', end='')

            # criar nova geração
            offspring = []
            for sub_set in itertools.combinations(ref_set, subset_size):
                s = self._recombination(sub_set)
                offspring.append(s)

            # atualizar conjunto de referência
            ref_set = self._ref_set_update(ref_set + offspring, ref_size, diver)

            # mutação
            for k in range(1, len(ref_set)):
                if rd.random() < mut_factor:
                    print('\nMutação')
                    cost, route = self._shake(ref_set[k][1], ref_set[k][0], k=2)
                    cost, route = self.VND(route)
                    ref_set[k] = cost, route
            ref_set.sort()

            # atualizar melhor solução encontrada
            cost, route = ref_set[0]
            if cost < best_cost:
                best_cost, best_route = deepcopy(ref_set[0])
                print(f'\n{i + 1} SS  {best_cost} ')
                if self.plot:
                    self.cvrp.plot(routes=best_route, clear_edges=True, stop=False)

        return best_cost, best_route

    def _ant_run(self, trail):

        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q
        c = self.cvrp.c
        sol = []

        visited = np.zeros([n], dtype=bool)

        cont = 1
        while cont < n:
            path = [0]
            v = 0
            load = 0
            while True:
                can = [i for i in range(n) if not visited[i] and load + d[i] <= q and v != i]
                if len(can) == 0:
                    break
                weight = [max(trail[v, i], self._min_trail) for i in can]

                # heuristic
                # weight = weight + np.array([1 / (c[v, i] + self._min_trail) for i in can])

                v = rd.choices(can, weights=weight)[0]
                if v == 0:
                    if load < .7 * q and len(can) > 1:
                        continue
                    else:
                        break
                else:
                    path.append(v)
                    load += d[v]
                    visited[v] = True
                    cont += 1
            sol.append(path)

        return sol

    _min_trail = 0.001

    def _reinforcement(self, sol, valor, trail):
        for r in sol:
            for i in range(1, len(r)):
                trail[r[i - 1], r[i]] += valor
            trail[r[-1], r[0]] += valor

    @timeit
    def ant_colony(self, ite: int, ants: int, evapor=0.1, online=True, update_by='quality', k=1):
        """
        Ant Colony Optimization

        :param ite: número de iterações
        :param ants: número de formigas
        :param evapor: taxa de evaporação
        :param online:
            True - a trilha é atualizada quando cada formiga termina seu percurso (Online delayed pheromone update);
            False - a trilha é atualizada apenas após todas as formigas terminarem seu percurso (offline)
        :param update_by:
            Usado quando online == False
            'quality' - as formigas que geraram as k melhores soluções depositam um valor constante às respectivas trilhas.
            'rank' - as formigas que geraram as k melhores soluções depositam um valor relativo as seu rank às respectivas trilhas.
            'worst' - a formiga que gerou a pior solução decrementa o feromônio  da sua trilha
            'elitist' - a melhor solução até então gerada adiciona feromônio à sua trilha
        :return:tupla (custo, solução)
        """
        n = self.cvrp.n
        trail = np.zeros(shape=[n, n], dtype=float)
        best_route = None
        best_cost = np.inf

        if online:
            # online delayed update
            best_cost, best_route = self.Clarke_n_Wright()
            print(f'\n{0} AC  {best_cost} ')
            UB = best_cost * 1.1
            for i in range(ite):
                for f in range(ants):
                    sol = self._ant_run(trail)
                    cost = self.cvrp.route_cost(sol)
                    print(cost)
                    cost, sol = self.VND(sol, cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = deepcopy(sol)
                        print(f'\n{i + 1} AC  {best_cost} ')
                    # evaporação
                    trail += (1 - evapor) * trail
                    # reforço
                    delta = (UB - cost) / UB
                    if delta > 0:
                        self._reinforcement(sol, delta, trail)

        # else:
        #     # offline update
        #     trail_aux = np.zeros(shape=[n, n], dtype=float)
        #     for i in range(ite):
        #         trail_aux.fill(0)
        #         for f in range(ants):
        #             sol = self._ant_run(trail)
        #             cost = self.cvrp.route_cost(sol)
        #             cost, sol = self.VND(sol, cost)
        #
        #             if cost < best_cost:
        #                 best_cost = cost
        #                 best_route = deepcopy(sol)
        #                 print(f'\n{i + 1} AC  {best_cost} ')
        #
        #             self._reinforcement(sol, cost - 1763, trail_aux)
        #         trail = 0.5 * trail + 0.5 * trail_aux

        return best_cost, best_route
