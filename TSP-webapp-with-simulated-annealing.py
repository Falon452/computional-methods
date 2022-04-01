from math import sqrt, exp
from random import randint
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# Run and go to http://127.0.0.1:8050/ #

# if error: import name 'get_current_traceback' from 'werkzeug.debug.tbtools'
# do : 1. pip uninstall werkzeug
#      2. pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.1.tar.gz


# -------- globals
X = []
Y = []
N_POINTS = 12
N_GROUPS = 3
N_EPOCHS = 100
N_ITERATIONS = 4
T_MULTIPLIER = 0.9
TEMPERATURE = 800
IS_NORMAL_DIST = True
IS_ARBITRARY_SWAP = False

colors = {
    'background': '#111111',
    'text': '#7FD8FF'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


# --------


def generate_points_uniform_distribution(n, low=0, high=10):
    xy_min = [0, 0]
    xy_max = [10, 20]
    data = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def generate_points_normal_dist(n, mean=None, covariance=None):
    if n < 0:
        print("generate_points_normal_dist: n must be >= 0.")
        exit(1)

    if not mean:
        mean = [5, 5]
    if not covariance:
        covariance = [[10, 0],
                      [0, 10]]

    x, y = np.random.multivariate_normal(mean, covariance, n).T
    return x, y


def generate_groups(points_in_group, n_groups, means=None, covariances=None):
    if not means:
        means = [[randint(0, 100), randint(0, 100)] for _ in range(n_groups)]
    if not covariances:
        covariances = [[[randint(1, 20), randint(1, 20)],
                        [randint(1, 20), randint(1, 20)]] for _ in range(n_groups)]
    if len(means) != n_groups:
        print("generate_groups:  len(means) != n_groups")
        exit(1)
    if len(covariances) != n_groups:
        print("generate_groups:  len(covariances) != n_groups")
        exit(1)
    if points_in_group < 0:
        print("generate_groups:  n_points < 0")
        exit(1)

    X = []
    Y = []
    for mean, covariance in zip(means, covariances):
        x, y = generate_points_normal_dist(points_in_group, mean, covariance)
        X.extend(x)
        Y.extend(y)
    return X, Y


def euclidean_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def global_distance(cities, x, y, distance_function=euclidean_distance):
    """
    :param distance_function: takes two points returns distance
    :param cities: indexes of points, the road
    :param x: list of x coordinates
    :param y: list of y coordinates
    :return:
    """
    distance = 0
    n = len(cities)
    for k in range(1, n):
        i = cities[k]
        j = cities[k - 1]
        x1, y1 = x[j], y[j]
        x2, y2 = x[i], y[i]
        a = (x1, y1)
        b = (x2, y2)
        distance += distance_function(a, b)

    x1, y1 = x[cities[n - 1]], y[cities[n - 1]]
    x2, y2 = x[cities[0]], y[cities[0]]
    a = (x1, y1)
    b = (x2, y2)
    distance += distance_function(a, b)
    return distance


def consecutive_swap(cities, at):
    n = len(cities)
    if at < 0 or at >= n:
        print("consecutive_swap provided at is not in range(0, n)")
        exit(1)

    if at == n - 1:
        cities[0], cities[n - 1] = cities[n - 1], cities[0]
    else:
        cities[at], cities[at + 1] = cities[at + 1], cities[at]


def arbitrary_swap_indices(cities):
    n = len(cities)
    if n <= 1:
        print("arbitrary_swap: can't swap one or none elements.")
        exit(1)

    i1 = randint(0, n - 1)
    i2 = i1
    while i2 == i1:
        i2 = randint(0, n - 1)
    return i1, i2


def accept_worse(prev, next, T):
    if next <= prev:
        print("accept: next <= prev")
        exit(1)

    probability = exp((prev - next) / T)
    if np.random.ranf() < probability:
        return True
    return False


def simulated_annealing(X, Y):
    cities = np.random.permutation(len(X))
    distances = []
    T_values = []

    T = TEMPERATURE

    counter_consecutive_swap = 0
    for _ in range(N_EPOCHS):
        for _ in range(N_ITERATIONS):
            previous_distance = global_distance(cities, X, Y, euclidean_distance)
            distances.append(previous_distance)
            T_values.append(T)

            if IS_ARBITRARY_SWAP:
                i, j = arbitrary_swap_indices(cities)
                cities[i], cities[j] = cities[j], cities[i]
            else:
                i = counter_consecutive_swap
                j = i + 1
                if i == len(X) - 1:
                    j = 0
                consecutive_swap(cities, counter_consecutive_swap)
                counter_consecutive_swap += 1
                counter_consecutive_swap %= (len(X))

            next_distance = global_distance(cities, X, Y, euclidean_distance)

            if next_distance > previous_distance:
                if accept_worse(previous_distance, next_distance, T):
                    continue
                else:
                    cities[i], cities[j] = cities[j], cities[i]

        T *= T_MULTIPLIER

    return cities, distances, T_values


def main():
    global X, Y, N_POINTS, N_GROUPS, N_EPOCHS, N_ITERATIONS, T_MULTIPLIER, IS_NORMAL_DIST, IS_ARBITRARY_SWAP, TEMPERATURE

    if IS_NORMAL_DIST:
        X, Y = generate_groups(N_POINTS // N_GROUPS, N_GROUPS)
    else:
        X, Y = generate_points_uniform_distribution(N_POINTS)

    cities, distances, T_values = simulated_annealing(X, Y)

    colors = {
        'background': '#111111',
        'text': '#7FD8FF'
    }

    df_points = pd.DataFrame({
        "x": X,
        "y": Y
    })
    fig_points = px.scatter(df_points, x="x", y="y")
    fig_points.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    fig_points.update_traces(marker=dict(size=12,
                                         line=dict(width=2,
                                                   color=colors['text'])),
                             selector=dict(mode='markers'))

    fig_points_sol = px.scatter(df_points, x="x", y="y")
    fig_points_sol.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )
    fig_points_sol.update_traces(marker=dict(size=8,
                                             ),
                                 opacity=0.7,
                                 selector=dict(mode='markers'))
    for i in range(1, len(cities)):
        x1, y1 = X[cities[i - 1]], Y[cities[i - 1]]
        x2, y2 = X[cities[i]], Y[cities[i]]

        arrow = dict(
            x=x2,
            y=y2,
            xref="x", yref="y",
            text="",
            showarrow=True,
            axref="x", ayref='y',
            ax=x1,
            ay=y1,
            arrowhead=1,
            arrowwidth=2,
            arrowcolor=colors['text']
        )
        fig_points_sol.add_annotation(arrow)

    x1, y1 = X[cities[-1]], Y[cities[-1]]
    x2, y2 = X[cities[0]], Y[cities[0]]

    arrow = dict(
        x=x2,
        y=y2,
        xref="x", yref="y",
        text="",
        showarrow=True,
        axref="x", ayref='y',
        ax=x1,
        ay=y1,
        arrowhead=1,
        arrowwidth=2,
        arrowcolor=colors['text'])
    fig_points_sol.add_annotation(arrow)

    df_distances = pd.DataFrame({
        "x": range(len(distances)),
        "distance": distances,
    })
    fig_distances = px.scatter(df_distances, x='x', y="distance")
    fig_distances.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )
    fig_distances.update_traces(marker=dict(size=8,
                                            line=dict(width=1.4,
                                                      color=colors['text'])),
                                selector=dict(mode='markers'))

    df_temp = pd.DataFrame({
        "x": range(len(distances)),
        "temp": T_values
    })
    fig_temperature = px.scatter(df_temp, x="x", y="temp", color="temp", color_continuous_scale="reds")
    fig_temperature.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )

    app.layout = html.Div(style={
        'backgroundColor': colors['background'],
        'display': 'flex', 'flex-direction': 'column', 'margin': 0, "margin-left": "0vw"

    }, children=[
        html.H1(
            children='TSP optimization using simulated annealing.',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'padding': 10, 'flex': 1
            }
        ),

        html.Div(children=[

            html.Div(children=[
                html.Label("Type of points", style={'color': colors['text'], 'font-size': 20, 'margin-left': 10}),
                dcc.RadioItems(["Normal Distribution", "Uniform Distribution"], 'Normal Distribution',
                               labelStyle={'display': 'block'}, style={'margin-left': 5, 'font-size': 16},
                               id='points_type')
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("Number of points", style={'margin-left': 35, 'font-size': 20}),
                dcc.Input(value=N_POINTS, type='text', style={'margin-left': 35, 'font-size': 16}, id='n_points')
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("Number of groups", style={'margin-left': 35, 'font-size': 20}),
                dcc.Input(value=N_GROUPS, type='text', style={'margin-left': 35, 'font-size': 16}, id='n_groups')
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("Type of swap", style={'color': colors['text'], 'margin-left': 40, 'font-size': 20}),
                dcc.RadioItems(["Arbitrary swap", 'Consecutive swap'], 'Arbitrary swap',
                               labelStyle={'display': 'block'},
                               style={'margin-left': 35, 'font-size': 16}, id='swap_type'),
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("Epochs (iterations to multiply T)", style={'margin-left': 35, 'font-size': 20}),
                dcc.Input(value=N_EPOCHS, type='text', style={'margin-left': 35, 'font-size': 16}, id='n_epochs')
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("Iterations in epoch", style={'margin-left': 35, 'font-size': 20}),
                dcc.Input(value=N_ITERATIONS, type='text', style={'margin-left': 35, 'font-size': 16},
                          id='n_iterations')
            ]),

            html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
                html.Label("T multiplier", style={'margin-left': 35, 'font-size': 20}),
                dcc.Input(value=T_MULTIPLIER, type='text', style={'margin-left': 35, 'font-size': 16},
                          id='T_multiplier')
            ])

        ], style={
            'textAlign': 'left',
            'padding': 30,
            'margin-left': 20,
            'color': colors['text'],
            'display': 'flex',
            'flex-direction': 'rows'}
            , ),

        html.Div(children=[
            html.Label("Initial Temperature", style={'color': colors['text'], 'margin-left': 30, 'font-size': 24,
                                                     'textAlign': 'center'}),
            html.Br(),
            dcc.Slider(
                min=0,
                max=5000,
                marks={
                    0: {'label': '0°C', 'style': {'color': '#FFFAFA', 'font-size': 18}},
                    250: {'label': '250°C', 'style': {'color': '#FAC000', 'font-size': 18}},
                    500: {'label': '500°C', 'style': {'color': '#FF7500', 'font-size': 18}},
                    750: {'label': '750°C', 'style': {'color': '#FC6400', 'font-size': 18}},
                    1000: {'label': '1000°C', 'style': {'color': '#FC6400', 'font-size': 18}},
                    1500: {'label': '1500°C', 'style': {'color': '#D73502', 'font-size': 18}},
                    2000: {'label': '2000°C', 'style': {'color': '#952B32', 'font-size': 18}},
                    3000: {'label': '3000°C', 'style': {'color': '#952B32', 'font-size': 18}},
                    4000: {'label': '4000°C', 'style': {'color': '#952B32', 'font-size': 18}},
                    5000: {'label': '5000°C', 'style': {'color': '#B62203', 'font-size': 18}}
                },
                value=700,
                id='temperature'
            ),
        ]),

        html.Div(children=[
            dcc.Graph(
                id='points',
                figure=fig_points,
                style={'flex': 1}
            ),

            dcc.Graph(
                id='tsp',
                figure=fig_points_sol,
                style={'flex': 1}
            ),
        ],
            style={'padding': 10, 'flex': 1, 'flex-direction': 'row', 'display': 'flex'}
        ),

        html.Div(children=[
            dcc.Graph(
                id='distance',
                figure=fig_distances,
                style={'height': '40vh'}
            ),

            dcc.Graph(
                id='temp',
                figure=fig_temperature,
                style={'height': '40vh'}
            )
        ],
            style={'padding': 0, 'flex': 2}
        )
    ])


@app.callback(
    Output('points', 'figure'),
    Output('tsp', 'figure'),
    Output('distance', 'figure'),
    Output('temp', 'figure'),
    Input('points_type', 'value'),
    Input('swap_type', 'value'),
    Input('n_points', 'value'),
    Input('n_groups', 'value'),
    Input('n_epochs', 'value'),
    Input('n_iterations', 'value'),
    Input('T_multiplier', 'value'),
    Input('temperature', 'value'),
)
def update_graphs(points_type, swap_type, n_points, n_groups, n_epochs, n_iter, T_multiplier, temperature):
    global X, Y, N_POINTS, N_GROUPS, N_EPOCHS, N_ITERATIONS, T_MULTIPLIER, IS_NORMAL_DIST, IS_ARBITRARY_SWAP, TEMPERATURE

    generate_new_points = False
    if points_type == 'Normal Distribution':
        if not IS_NORMAL_DIST:
            IS_NORMAL_DIST = True
            generate_new_points = True
    else:
        if IS_NORMAL_DIST:
            IS_NORMAL_DIST = False
            generate_new_points = True

    if swap_type == 'Arbitrary swap':
        if not IS_ARBITRARY_SWAP:
            IS_ARBITRARY_SWAP = True
    else:
        if IS_ARBITRARY_SWAP:
            IS_ARBITRARY_SWAP = False

    if n_points != '':
        n_points = int(n_points)
        if n_points != 0:
            if N_POINTS != n_points:
                N_POINTS = n_points
                generate_new_points = True

    if n_groups != '':
        n_groups = int(n_groups)
        if n_groups != 0:
            if N_GROUPS != n_groups:
                N_GROUPS = n_groups
                generate_new_points = True

    if n_epochs != '':
        n_epochs = int(n_epochs)
        if n_epochs != 0:
            if N_EPOCHS != n_epochs:
                N_EPOCHS = n_epochs

    if n_iter != '':
        n_iter = int(n_iter)
        if n_iter != 0:
            if N_ITERATIONS != n_iter:
                N_ITERATIONS = n_iter

    if T_multiplier != '':
        T_multiplier = float(T_multiplier)
        if T_MULTIPLIER != T_multiplier:
            T_MULTIPLIER = T_multiplier

    if temperature != '':
        temperature = int(temperature)
        if TEMPERATURE != temperature:
            TEMPERATURE = temperature

    if generate_new_points:
        if IS_NORMAL_DIST:
            X, Y = generate_groups(N_POINTS // N_GROUPS, N_GROUPS)
        else:
            X, Y = generate_points_uniform_distribution(N_POINTS)

    cities, distances, T_values = simulated_annealing(X, Y)

    df_points = pd.DataFrame({
        "x": X,
        "y": Y
    })
    fig_points = px.scatter(df_points, x="x", y="y")

    fig_points.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    fig_points.update_traces(marker=dict(size=12,
                                         line=dict(width=2,
                                                   color=colors['text'])),
                             selector=dict(mode='markers'))

    fig_points_sol = px.scatter(df_points, x="x", y="y")

    fig_points_sol.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )

    fig_points_sol.update_traces(marker=dict(size=10,
                                             line=dict(width=2,
                                                       color=colors['text']),
                                             opacity=0.6
                                             ),

                                 selector=dict(mode='markers'))

    for i in range(1, len(cities)):
        x1, y1 = X[cities[i - 1]], Y[cities[i - 1]]
        x2, y2 = X[cities[i]], Y[cities[i]]

        arrow = dict(
            x=x2,
            y=y2,
            xref="x", yref="y",
            text="",
            showarrow=True,
            axref="x", ayref='y',
            ax=x1,
            ay=y1,
            arrowhead=1,
            arrowwidth=2,
            arrowcolor=colors['text']
        )
        fig_points_sol.add_annotation(arrow)

    x1, y1 = X[cities[-1]], Y[cities[-1]]
    x2, y2 = X[cities[0]], Y[cities[0]]

    arrow = dict(
        x=x2,
        y=y2,
        xref="x", yref="y",
        text="",
        showarrow=True,
        axref="x", ayref='y',
        ax=x1,
        ay=y1,
        arrowhead=1,
        arrowwidth=2,
        arrowcolor=colors['text'])

    fig_points_sol.add_annotation(arrow)

    df_distances = pd.DataFrame({
        "x": range(len(distances)),
        "distances": distances,
    })

    fig_distances = px.scatter(df_distances, x="x", y="distances")

    fig_distances.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )

    fig_distances.update_traces(marker=dict(size=8,
                                            line=dict(width=1.4,
                                                      color=colors['text'])),
                                selector=dict(mode='markers'))

    df_temp = pd.DataFrame({
        "x": range(len(distances)),
        "temp": T_values
    })

    fig_temperature = px.scatter(df_temp, x="x", y="temp", color="temp", color_continuous_scale="reds")
    fig_temperature.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'showgrid': False},
        yaxis={'showgrid': False}
    )

    return fig_points, fig_points_sol, fig_distances, fig_temperature


if __name__ == '__main__':
    main()
    app.run_server(debug=True)
