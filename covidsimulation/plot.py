from functools import partial
from typing import Sequence, Tuple, Optional, Union

import numpy as np
import plotly.graph_objects as go

from .series import Series
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


def plot_line(fig, series, pop_name, color_index):
    fig.add_trace(go.Scatter(
        x=series.x,
        y=series.y,
        line_color=PLOT_COLORS[color_index][0],
        name=pop_name,
    ))


def concat_seq(s1, s2):
    if isinstance(s1, np.ndarray):
        return np.stack([s1, s2]).reshape(len(s1) + len(s2))
    return list(s1) + list(s2)


def plot_confidence_range(fig, series_low, series_high, legend, color_index):
    assert len(series_low.x) == len(series_high.x)
    fig.add_trace(go.Scatter(
        x=concat_seq(series_high.x, series_low.x[::-1]),
        y=concat_seq(series_high.y, series_low.y[::-1]),
        fill='toself',
        fillcolor=PLOT_COLORS[color_index][1],
        line_color=PLOT_COLORS[color_index][1],
        showlegend=legend is not None,
        name=legend,
    ))


def plot(
        pop_stats_name_tuples: Sequence[Tuple[Union[MetricResult, Series, Sequence[Series]], str]],
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
        if isinstance(data, Series):
            line_fn = partial(plot_line, fig, data.trim(start, stop), pop_name, color_index)
            line_fns.append(line_fn)
        elif len(data) == 1:
            line_fn = partial(plot_line, fig, data[0].trim(start, stop), pop_name, color_index)
            line_fns.append(line_fn)
        elif len(data) == 3:
            if show_confidence_interval:
                area_fn = partial(plot_confidence_range, fig, data[1].trim(start, stop), data[2].trim(start, stop),
                                  None, color_index)
                area_fns.append(area_fn)
            line_fn = partial(plot_line, fig, data[0].trim(start, stop), pop_name, color_index)
            line_fns.append(line_fn)
        elif len(data) == 2:
            area_fn = partial(plot_confidence_range, fig, data[0].trim(start, stop), data[1].trim(start, stop),
                              pop_name, color_index)
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
