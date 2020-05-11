import datetime
from copy import deepcopy
from functools import partial
from typing import Sequence, Tuple, Optional, Union

import numpy as np
import plotly.graph_objects as go

from .utils import get_date_from_isoformat

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
                start_date = get_date_from_isoformat(start_date)
            self.x = [start_date + datetime.timedelta(days=i) for i in range(len(self.y))]
        else:
            self.x = x

    def __getitem__(self, item):
        item = self.get_index(item)
        return self.y[item]

    def __len__(self):
        return len(self.y)

    @property
    def start_date(self):
        return self.x[0]

    def get_index(self, x_value):
        if isinstance(x_value, str):
            x_value = get_date_from_isoformat(x_value)
        if isinstance(x_value, datetime.date):
            x_value = (x_value - self.start_date).days
        return x_value

    def tolist(self):
        return self.y.tolist()


def plot_line(fig, series, pop_name, color_index, stop, start):
    start_index = series.get_index(start) if start else 0
    stop_index = series.get_index(stop) if stop else len(series)
    fig.add_trace(go.Scatter(
        x=series.x[start_index:stop_index],
        y=series.y[start_index:stop_index],
        line_color=PLOT_COLORS[color_index][0],
        name=pop_name,
    ))


def concat_seq(s1, s2):
    if isinstance(s1, np.ndarray):
        return np.stack([s1, s2]).reshape(len(s1) + len(s2))
    return list(s1) + list(s2)


def plot_confidence_range(fig, series_low, series_high, legend, color_index, stop, start):
    assert series_low.x == series_high.x
    start_index = series_low.get_index(start) if start else 0
    stop_index = series_low.get_index(stop) if stop else len(series_low)
    plot_indices = list(range(start_index, stop_index))
    fig.add_trace(go.Scatter(
        x=concat_seq(series_high.x[plot_indices], series_low.x[reversed(plot_indices)]),
        y=concat_seq(series_high.y[plot_indices], series_low.y[reversed(plot_indices)]),
        fill='toself',
        fillcolor=PLOT_COLORS[color_index][1],
        line_color=PLOT_COLORS[color_index][1],
        showlegend=legend is not None,
        name=legend,
    ))


def plot(
        pop_stats_name_tuples: Sequence[Tuple[Union['MetricResult', Series, Sequence[Series]], str]],
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
    area_fns = []
    line_fns = []

    for color_index, (data, pop_name) in enumerate(pop_stats_name_tuples):
        color_index = color_index % len(PLOT_COLORS)
        if cindex is not None:
            color_index = cindex
        if isinstance(data, Series) or len(data) == 1:
            line_fn = partial(plot_line, fig, data[0], pop_name, color_index, stop, start)
            line_fns.append(line_fn)
        elif len(data) == 3:
            if show_confidence_interval:
                area_fn = partial(plot_confidence_range, fig, data[1], data[2], None, color_index, stop, start)
                area_fns.append(area_fn)
            line_fn = partial(plot_line, fig, data[0], pop_name, color_index, stop, start)
            line_fns.append(line_fn)
        elif len(data) == 2:
            area_fn = partial(plot_confidence_range, fig, data[0], data[1], pop_name, color_index, stop, start)
            area_fns.append(area_fn)
        else:
            raise ValueError('Invalid number of elements to plot')

    for area_fn in area_fns:
        area_fn()

    for line_fn in line_fns:
        line_fn()

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
