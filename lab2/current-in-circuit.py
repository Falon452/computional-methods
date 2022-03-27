import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx import cycle_basis
from decimal import Decimal

"""
1. For every node make equation from I Kirchhoff's law
2. Find simple cycles in graph:
Used cycle_basis networkx method that uses CACM Algorithm 491
http://fatphil.org/maths/resources/Gibbs_BasicCycleGeneration.html
3. For every cycle make equation from II Kirchoff's law
The resulting matrix will be over-determined
4. Solve using numpy least square solution

Assert if I Kirchhoff's law is satisfied

Draw pretty graph
"""


def read_graph_from_list_of_edges(path):
    """
    expected format:
    (from, to, weight), (1, 3, 636), (0, 1, 1)
    from, to, SEM

    :param path:
    :return:
    """
    with open(path, 'r') as f:
        edges = f.readline()
        edges = edges.strip()
        edges = edges[1:-1]
        edges = edges.split("), (")
        edges = [edge.split(", ") for edge in edges]
        edges = [[int(value) for value in edge] for edge in edges]
        stE = f.readline()
        stE = stE.split(", ")
        stE = list(map(int, stE))
        s, t, E = stE
        return edges, s, t, E


def read_graph_from_matrix(path):
    """
    expected format:
    0, 1, 2, 1
    1, 0, 1, 1
    2, 1, 0, 2
    1, 1, 2, 0
    0, 1, 10
    :param path:
    :return:
    """
    G = []
    with open(path, 'r') as f:
        row = f.readline()
        row = row.split(', ')
        row = list(map(int, row))
        G.append(row)
        n = len(row)
        for _ in range(1, n):
            row = list(map(int, f.readline().split(', ')))
            G.append(row)
        s, t, E = list(map(int, f.readline().split(", ")))
    return G, s, t, E


def convert_edges_to_matrix(edges):
    """
    converts from list of edges [(from, to, weight), ...] to matrix representation

    :param edges:
    :return:
    """
    n = 0
    for edge in edges:
        v, u, w = edge
        n = max(n, v, u)
    n += 1
    G = [[0 for _ in range(n)] for _ in range(n)]

    for edge in edges:
        v, u, w = edge
        G[v][u] = w
        G[u][v] = w
    return G


def get_erdos_renyi(n, p):
    return nx.erdos_renyi_graph(n, p)


def get_3regular_graph(n):
    return nx.random_regular_graph(3, n)


def make_graph_randomly_weighted(G, weigth_start, weight_end, attr_name="weight"):
    """
    void, adds weight to given graph G

    :param G: networkx Graph
    :param weigth_start: inclusive
    :param weight_end: exclusive
    :param attr_name:
    :return:
    """
    for u, v in G.edges():
        G[u][v][attr_name] = np.random.randint(weigth_start, weight_end)


def add_weight_to_edges(G, weights, attr_name="current"):
    # G is a networkx graph
    for i, (u, v) in enumerate(G.edges()):
        G[u][v][attr_name] = weights[i]


def add_indices_to_edges(G, attr_name="index"):
    # G is a networkx graph
    for i, (u, v) in enumerate(G.edges()):
        G[u][v][attr_name] = i


def make_graph_with_bridge(n1, p1, n2, p2):
    """
    makes two random erdos renyi graphs, and connect them with bridge

    :param n1:
    :param p1:
    :param n2:
    :param p2:
    :return: networkx Graph, bridge_from, bridge_to
    """
    G1 = get_erdos_renyi(n1, p1)
    G2 = get_erdos_renyi(n2, p2)

    bridge_v, bridge_u = None, None
    for v, _ in G1.edges():
        bridge_v = v
        break
    for _, u in G2.edges():
        bridge_u = u
        break

    assert bridge_v is not None and bridge_u is not None, "bridge is not connected"

    bridge_u += n1
    G1.add_edge(bridge_v, bridge_u)
    print(f"Bridge is {bridge_v} -> {bridge_u}")

    for u, v in G2.edges():
        u += n1
        v += n1
        G1.add_edge(u, v)

    return G1, bridge_v, bridge_u


def make_graph_small_world(n, k=4, p=0.5, tries=20, print_coefficients=True):
    G = nx.connected_watts_strogatz_graph(n=n, k=k, p=p, tries=tries)
    if print_coefficients:  # takes a lot of time
        print(f"Small-world coefficient (sigma) : {nx.sigma(G)}\n "  # Sigma > 1  then network is small-world.
              f"Small-world coefficient(omega) : {nx.omega(G)}")  # Values close to 0 then network is more small-world.
    return G


def draw_and_save(G, filename, SEM, edge_color_attr="current", edge_weight_attr="resistance", sem_edge=(None, None),
                  bridge_edge=(None, None), grid_layout=False, current_label=False):
    """
    For grid2D graph it will draw edges with negative current, meaning the current actually flows in opposite direction

    :param G: networkx graph
    :param filename: where to save drawed graph
    :param SEM: number of voltage on sem_edge
    :param edge_color_attr: string, G.networkx attribute
    :param edge_weight_attr: string, G.networkx attribute
    :param sem_edge: (from, to)
    :param bridge_edge: if given, it will label it
    :param grid_layout: if we want Grid2D
    :param current_label: if we want to see what exactly current flows through edge
    :return:
    """
    n = len(G.nodes())
    NODE_COLOR = "lightblue"
    NODE_SIZE = 700
    NODE_FONT_SIZE = 20
    NODE_LABEL_ALPHA = 0.7
    ALLOW_ARROWS = True
    ARROW_SIZE = 120
    CMAP = plt.cm.plasma
    EDGE_WIDTH = 6
    EDGE_ALPHA = 0.8
    EDGE_FONT_COLOR = "red"
    EDGE_FONT_SIZE = 35
    CBAR_TICK_SIZE = 120
    BACKGROUND = "darkgrey"
    FIG_WIDTH = 60
    FIG_HEIGHT = 40
    FIG_DPI = 50
    CURRENT_LABEL_COLOR = "green"
    CURRENT_POS = 0.3
    SEM_COLOR = "blue"
    SEM_POS = 0.7
    K = 1
    SEM_FONT_SIZE = 40

    if n < 15:
        NODE_COLOR = "lightblue"
        NODE_SIZE = 2200
        NODE_FONT_SIZE = 50
        ARROW_SIZE = 100
        EDGE_WIDTH = 12
        EDGE_ALPHA = 1
        EDGE_FONT_COLOR = "red"
        EDGE_FONT_SIZE = 35
        CBAR_TICK_SIZE = 120
        BACKGROUND = "grey"
        FIG_WIDTH = 60
        FIG_HEIGHT = 40
        K = 1

    elif 15 <= n < 50:
        NODE_SIZE = 1000
        NODE_FONT_SIZE = 15
        ARROW_SIZE = 50
        EDGE_WIDTH = 5
        EDGE_FONT_SIZE = 30
        K = 2
        FIG_WIDTH = 80
        FIG_HEIGHT = 60
        FIG_DPI = 60
    elif n >= 50:
        CMAP = plt.cm.viridis

        NODE_SIZE = 800
        NODE_FONT_SIZE = 12
        ARROW_SIZE = 30
        EDGE_WIDTH = 2
        EDGE_FONT_SIZE = 0
        EDGE_ALPHA = 0.6
        K = 2
        FIG_WIDTH = 80
        FIG_HEIGHT = 60
        FIG_DPI = 60

    if edge_color_attr and edge_weight_attr:
        edges, edge_colors = zip(*nx.get_edge_attributes(G, edge_color_attr).items())
        edges, edge_width = zip(*nx.get_edge_attributes(G, edge_weight_attr).items())

        fig = plt.figure(1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
        ax = plt.gca()

        if grid_layout:
            pos = nx.spring_layout(G, iterations=100, seed=39775)
        else:
            pos = nx.spring_layout(G, k=K, iterations=100, seed=39775)

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=NODE_COLOR,
            node_size=NODE_SIZE
        )

        edges = nx.draw_networkx_edges(
            G,
            pos,
            arrows=ALLOW_ARROWS,
            arrowsize=ARROW_SIZE,
            edge_color=edge_colors,
            edge_cmap=CMAP,
            width=EDGE_WIDTH,
            alpha=EDGE_ALPHA
        )

        node_labels = nx.draw_networkx_labels(
            G,
            pos,
            font_size=NODE_FONT_SIZE,
            alpha=NODE_LABEL_ALPHA
        )

        edge_labels = nx.get_edge_attributes(G, edge_weight_attr)

        formatted_edge_labels = {(elem[0], elem[1]): str(round(edge_labels[elem], 2)) + ' \u03A9' for elem in
                                 edge_labels}

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=formatted_edge_labels,
            font_color=EDGE_FONT_COLOR,
            font_size=EDGE_FONT_SIZE,
            alpha=1,
            label_pos=0.6
        )

        if sem_edge[0] is not None and sem_edge[1] is not None:
            sem_label = {(sem_edge[0], sem_edge[1]): f"SEM={SEM} V"}
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=sem_label,
                font_color=SEM_COLOR,
                font_size=SEM_FONT_SIZE,
                label_pos=SEM_POS,
                alpha=1
            )

        if bridge_edge[0] is not None and bridge_edge[1] is not None:
            edge_data = G.edges.get(bridge_edge)
            if edge_data is None:  # the current was in the opposite direction
                edge_data = G.edges.get((bridge_edge[1], bridge_edge[0]))

            b_r = edge_data["resistance"]
            b_c = edge_data["current"]

            bridge_label = {(bridge_edge[0], bridge_edge[1]): f"BRIDGE C: {'%.2E' % Decimal(b_c)} R: {b_r}"}

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=bridge_label,
                font_color="brown",
                font_size=SEM_FONT_SIZE,
                label_pos=SEM_POS,
                alpha=1
            )

        if current_label:
            edge_labels_current = nx.get_edge_attributes(G, edge_color_attr)
            formatted_edge_labels_current = {(elem[0], elem[1]): str(round(edge_labels_current[elem], 2)) + ' A' for
                                             elem in
                                             edge_labels_current}
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=formatted_edge_labels_current,
                font_color=CURRENT_LABEL_COLOR,
                font_size=EDGE_FONT_SIZE,
                alpha=1,
                label_pos=CURRENT_POS
            )

        ax1_divider = make_axes_locatable(ax)
        cax = ax1_divider.append_axes("right", size="7%", pad="2%")
        cax.set_axis_on()

        mappable = cm.ScalarMappable(norm=Normalize(min(edge_colors), max(edge_colors)), cmap=CMAP)
        mappable.set_array(edge_colors)
        fig.colorbar(mappable, cax=cax, orientation='vertical')

        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=CBAR_TICK_SIZE)

        ax.set_facecolor(BACKGROUND)

        plt.savefig(filename)


def check_first_kirchhoff_law(G):
    """
    Expects edge[current] to be negative, ex. v -- -5A --> u means u -- 5A --> v

    :param G:
    :return:
    """
    edges_data = list(G.edges.data())
    for node in G.nodes():
        node_current = 0
        for ix, edge_data in enumerate(edges_data):
            u = edge_data[0]
            v = edge_data[1]
            current = edge_data[2]["current"]
            if u == node:  # u -> v
                node_current -= current
            if v == node:
                node_current += current
        if node_current > 1e-8:  # unsatisfied
            return False
    return True  # satisfied


def current_directed_graph(G):
    """
    Takes undirected graph with negative current, transfers it into directed graph with only
    positive current in right direction.
    :param G: networkx graph
    :return:
    """
    H = nx.DiGraph()
    edges_data = list(G.edges.data())
    for edge_data in edges_data:
        v = edge_data[0]
        u = edge_data[1]
        r = edge_data[2]["resistance"]
        c = edge_data[2]["current"]
        if c >= 0:
            H.add_edge(v, u, resistance=r, current=c)
        else:
            H.add_edge(u, v, resistance=r, current=-c)
    return H


def solve_for_current(G, SEM):
    """
    G undirected networkx graph (not Grid2D)

    1. For every node make equation from I Kirchhoff's law
    2. Find simple cycles in graph:
    Used cycle_basis networkx method that uses CACM Algorithm 491
    http://fatphil.org/maths/resources/Gibbs_BasicCycleGeneration.html
    3. For every cycle make equation from II Kirchoff's law
    The resulting matrix will be over-determined
    4. Solve using numpy least square solution


    :param G:
    :param SEM:
    :return: np.linalg.lstsq(A, B)
    """
    add_indices_to_edges(G)
    n = len(G.nodes)
    m = len(G.edges)

    A = [[0 for _ in range(m)] for _ in range(n)]
    B = [0 for _ in range(n)]
    edges_data = list(G.edges.data())
    for ix, edge_data in enumerate(edges_data):
        v = edge_data[0]
        u = edge_data[1]
        A[v][ix] = -1
        A[u][ix] = 1

    cycles = cycle_basis(G)

    for cycle in cycles:
        zeros = [0 for _ in range(m)]
        for i in range(len(cycle)):
            if i == len(cycle) - 2:
                v = cycle[-1]
                u = cycle[-2]
                r = G.edges.get((v, u))["resistance"]
                ix = G.edges.get((v, u))["index"]
                zeros[ix] = -r
                continue
            elif i == len(cycle) - 1:
                v = cycle[-1]
                u = cycle[0]
            else:
                v = cycle[i]
                u = cycle[i + 1]
            r = G.edges.get((v, u))["resistance"]
            ix = G.edges.get((v, u))["index"]
            zeros[ix] = r

        A.append(zeros)
        B.append(SEM)

    return np.linalg.lstsq(A, B)


def solve_for_grid_graph(G, SEM):
    """
    G - networkx grid graph2d - undirected

    The output will have negative current so if there is -5A from v -> u
    it means that there is 5A from u->v


    :param G:
    :param SEM:
    :return: G with current as data
    """
    T = nx.Graph()
    mapped_nodes = {}
    i = 0
    for node in G.nodes():
        mapped_nodes[node] = i
        i += 1
    for edge in G.edges.data():
        T.add_edge(mapped_nodes[edge[0]], mapped_nodes[edge[1]], resistance=edge[2]["resistance"])

    result = solve_for_current(T, SEM)

    add_weight_to_edges(T, result[0], "current")

    assert check_first_kirchhoff_law(T), "first kirchhoff law"

    for edge in list(T.edges.data()):
        u = edge[0]
        w = edge[1]
        for k, v in mapped_nodes.items():
            if u == v:
                unmapped_u = k
            if w == v:
                unmapped_w = k

        r = edge[2]["resistance"]
        c = edge[2]["current"]
        G[unmapped_u][unmapped_w]["current"] = c
        G[unmapped_u][unmapped_w]["resistance"] = r
    return G


def solve_and_ready_to_draw(G, SEM):
    result = solve_for_current(G, SEM)
    add_weight_to_edges(G, result[0], "current")
    assert check_first_kirchhoff_law(G)
    return current_directed_graph(G)


if __name__ == "__main__":
    b_v, b_u = None, None
    sem_v, sem_u = None, None
    SEM = 1000

    G = make_graph_small_world(40, print_coefficients=False)

    # dim1 = 7
    # dim2 = 5
    # G = nx.grid_2d_graph(dim1, dim2)
    make_graph_randomly_weighted(G, 1, 100, "resistance")
    G = solve_and_ready_to_draw(G, SEM)
    # G = solve_for_grid_graph(G, 100)

    # G, b_v, b_u = make_graph_with_bridge(100, 0.8, 100, 0.8)
    # make_graph_randomly_weighted(G, 1, 100, "resistance")
    # G = solve_and_ready_to_draw(G, SEM)
    nodes = len(G.nodes())

    # H = solve_and_ready_to_draw(G, SEM)

    if sem_v is None and sem_u is None:
        sem_v = list(G.edges().data())[0][0]
        sem_u = list(G.edges().data())[0][1]

    draw_and_save(G, f"./images/small-world-{nodes}", edge_color_attr="current", edge_weight_attr="resistance",
                  sem_edge=(sem_v, sem_u), SEM=SEM, bridge_edge=(b_v, b_u), grid_layout=False, current_label=False)
