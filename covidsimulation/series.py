import datetime
from copy import deepcopy
from typing import Optional, Union, Sequence, Tuple, Mapping

import numpy as np

from .utils import get_date_from_isoformat


class Series:
    x_date: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __init__(self, y: Union[Sequence, np.ndarray, Mapping], x: Optional[Union[Sequence, np.ndarray]] = None,
                 start_date: Optional[Union[str, datetime.date]] = None):
        if (x is None) and (start_date is None) and (not isinstance(y, Mapping)):
            raise ValueError('Either x or start_date must be specified or a mapping must be passed')
        if not((x is None) and (start_date is None)) and (isinstance(y, Mapping)):
            raise ValueError('Neither x or start_date must be specified when a mapping is passed')
        if isinstance(y, Mapping):
            x = list(y.keys())
            y = list(y.values())
        if not start_date:
            assert list(x) == sorted(list(x))
            x = make_date_sequence(x)
            start_date = x[0]
            end_date = x[-1]
            num_elements = (end_date - start_date).days + 1
        else:
            if isinstance(start_date, str):
                start_date = get_date_from_isoformat(start_date)
            num_elements = len(y)
        self.x_date = np.array([start_date + datetime.timedelta(days=i) for i in range(num_elements)])
        if num_elements > len(y):
            last = 0.0
            x_to_y = map_x_to_y(x, y)
            new_y = []
            for day in self.x_date:
                last = x_to_y.get(day, last)
                new_y.append(last)
            y = new_y
        self.x = np.array([to_datetime(d) for d in self.x_date])
        if isinstance(y, np.ndarray):
            self.y = deepcopy(y)
        else:
            self.y = np.array(y)

    def __getitem__(self, item):
        item = self.get_index(item)
        return self.y[item]

    def __len__(self):
        return len(self.y)

    def __add__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(a.y + b.y, a.x_date)
        elif isinstance(other, datetime.timedelta):
            return Series(self.y, self.x_date + other)
        else:
            return Series(self.y + other, self.x_date)

    def __radd__(self, other):
        if not other:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(a.y * b.y, a.x_date)
        else:
            return Series(self.y * other, self.x_date)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(a.y / (b.y + 0.000000001), a.x_date)
        else:
            return Series(self.y / other, self.x_date)

    def __rtruediv__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(b.y / (a.y + 0.000000001), a.x_date)
        else:
            return Series(other / self.y, self.x_date)

    def __sub__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(a.y - b.y, a.x_date)
        elif isinstance(other, datetime.timedelta):
            return Series(self.y, self.x_date - other)
        else:
            return Series(self.y - other, self.x_date)

    def __rsub__(self, other):
        if isinstance(other, Series):
            a, b = align_series(self, other)
            return Series(b.y - a.y, a.x_date)
        else:
            return Series(other - self.y, self.x_date)

    @property
    def start_date(self):
        return self.x_date[0]

    def get_index(self, x_value):
        if isinstance(x_value, str):
            x_value = get_date_from_isoformat(x_value)
        if isinstance(x_value, datetime.date):
            x_value = min(max((x_value - self.start_date).days, 0), len(self.x_date) - 1)
        return x_value

    def tolist(self):
        return self.y.tolist()

    def trim(self, start: Optional[Union[int, datetime.date, str]] = None,
             stop: Optional[Union[int, datetime.date, str]] = None):
        start_index = self.get_index(start or 0)
        stop_index = get_stop_index(self, stop)
        return Series(self.y[start_index:stop_index], self.x_date[start_index:stop_index])


def make_date_sequence(x: Sequence):
    x_date = []
    for day in x:
        if isinstance(day, datetime.datetime):
            day = day.isoformat()
        if isinstance(day, str):
            day = get_date_from_isoformat(day)
        x_date.append(day)
    return x_date


def map_x_to_y(x: Sequence, y: Sequence):
    assert len(x) == len(y)
    x_to_y = {}
    for xi, yi in zip(x, y):
        x_to_y[xi] = yi
    return x_to_y


def to_datetime(x: datetime.date) -> datetime.datetime:
    return datetime.datetime(*x.timetuple()[:3])


def get_stop_index(series, stop):
    stop_index = series.get_index(stop) + 1 if stop is not None else len(series)
    return min(stop_index, len(series))


def align_series(a: Series, b: Series) -> Tuple[Series, Series]:
    common_start = max(a.x_date[0], b.x_date[0])
    common_end = min(a.x_date[-1], b.x_date[-1])
    return a.trim(common_start, common_end), b.trim(common_start, common_end)
