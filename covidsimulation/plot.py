import datetime
from copy import deepcopy
from typing import Sequence, Tuple, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .stats import MetricResult

PLOT_COLORS = [
    ('rgba(0,0,255,1)', 'rgba(0,0,255,0.25)'),
    ('rgba(255,0,0,1)', 'rgba(255,0,0,0.25)'),
    ('rgba(0,0,0,1)', 'rgba(0,0,0,0.25)'),
    ('rgba(128,0,240,1)', 'rgba(128,0,240,0.25)'),
    ('rgba(240,128,0,1)', 'rgba(240,128,0,0.25)'),
    ('rgba(0,128,240,1)', 'rgba(0,128,240,0.25)'),
    ('rgba(0,255,0,1)', 'rgba(0,255,0,0.25)'),
]


class Series:
    x: Sequence[datetime.date]
    y: np.ndarray

    def __init__(self, y: np.ndarray, x: Optional[Sequence] = None,
                 start_date: Optional[Union[str, datetime.date]] = None):
        self.y = deepcopy(y)
        if (x is None) and (start_date is None):
            raise ValueError('Either x or start_date must be specified')
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.date.fromisoformat(start_date)
            self.x = [start_date + datetime.timedelta(days=i) for i in range(len(self.y))]
        else:
            self.x = x


def get_population_plot(fig, pop_stats: MetricResult, pop_name, color_index, stop, start, show_confidence_interval):
    days = [
        {'mean': x[0], 'min': x[1], 'max': x[2]} for x in zip(pop_stats.mean, pop_stats.low, pop_stats.high)
    ]
    if stop:
        days = days[:stop]
    if start:
        days = days[start:]
    df = pd.DataFrame(days)
    df['x'] = pd.date_range(pop_stats.stats.start_date, periods=len(days)).to_pydatetime()
    if start:
        df['x'] += datetime.timedelta(days=start)
    x = df['x']
    x_rev = x[::-1]
    x_plot = pd.concat([x, x_rev], ignore_index=True)
    y1 = df['mean']
    if show_confidence_interval:
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


def plot(
        pop_stats_name_tuples: Sequence[Tuple[MetricResult, str]],
        title: str,
        log_scale: bool = False,
        size: Optional[int] = None,
        stop: Optional[int] = None,
        start: Optional[int] = None,
        ymax: Optional[float] = None,
        cindex: Optional[int] = None,
        show_confidence_interval: bool = True,
):
    fig = go.FigureWidget()
    for color_index, (pop_stats, pop_name) in enumerate(pop_stats_name_tuples):
        color_index = color_index % len(PLOT_COLORS)
        if cindex is not None:
            color_index = cindex
        get_population_plot(fig, pop_stats, pop_name, color_index, stop, start, show_confidence_interval)
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
    return fig
