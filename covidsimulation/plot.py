import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOT_COLORS = [
    ('rgba(0,0,255,1)', 'rgba(0,0,255,0.25)'),
    ('rgba(255,0,0,1)', 'rgba(255,0,0,0.25)'),
    ('rgba(0,0,0,1)', 'rgba(0,0,0,0.25)'),
    ('rgba(128,0,240,1)', 'rgba(128,0,240,0.25)'),
    ('rgba(240,128,0,1)', 'rgba(240,128,0,0.25)'),
    ('rgba(0,128,240,1)', 'rgba(0,128,240,0.25)'),
    ('rgba(0,255,0,1)', 'rgba(0,255,0,0.25)'),
]


def get_population_plot(fig, pop_stats, pop_name, color_index, stop, start):
    stats, smean, smin, smax = pop_stats
    days = [
        {'mean': x[0], 'min': x[1], 'max': x[2]} for x in zip(smean, smin, smax)
    ]
    if stop:
        days = days[:stop]
    if start:
        days = days[start:]
    df = pd.DataFrame(days)
    df['x'] = pd.date_range(stats.start_date, periods=len(days)).to_pydatetime()
    if start:
        df['x'] += datetime.timedelta(days=start)
    x = df['x']
    x_rev = x[::-1]
    x_plot = pd.concat([x, x_rev], ignore_index=True)
    y1 = df['mean']
    y1_upper = df['max']
    y1_lower = df['min']
    y1_lower = y1_lower[::-1]
    y1_plot = pd.concat([y1_upper, y1_lower], ignore_index=True)
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y1_plot,
        fill='toself',
        fillcolor=PLOT_COLORS[color_index][1],
        line_color=PLOT_COLORS[color_index][1],
        showlegend=False,
        name=pop_name,
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y1,
        line_color=PLOT_COLORS[color_index][0],
        name=pop_name,
    ))


def plot(pop_stats_name_tuples, title, log_scale=False, size=None, stop=None, start=None, ymax=None, cindex=None):
    fig = go.Figure()
    for color_index, (pop_stats, pop_name) in enumerate(pop_stats_name_tuples):
        color_index = color_index % len(PLOT_COLORS)
        if not cindex is None:
            color_index = cindex
        get_population_plot(fig, pop_stats, pop_name, color_index, stop, start)
    fig.update_layout(
        title=title)
    if ymax:
        fig.update_yaxes(range=[1 if log_scale else 0, ymax])
    if log_scale:
        fig.update_layout(yaxis_type="log")
    if size:
        fig.update_layout(width=size)
    if len(pop_stats_name_tuples) == 1:
        fig.update_layout(showlegend=False)
    fig.show()
