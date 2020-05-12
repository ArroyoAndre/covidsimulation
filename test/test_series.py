import pytest

from datetime import date, timedelta

from covidsimulation.series import Series


def test_series():
    s1 = Series([1.0, 2.0, 3.0], start_date='2020-01-01')
    assert len(s1) == 3
    assert s1.x_date[-1] == date(2020, 1, 3)

    s2 = Series({'2020-01-02': 15.0, '2020-01-04': 18.0})
    assert len(s2) == 3
    assert s2.x_date[1] == date(2020, 1, 3)
    assert s2['2020-01-03'] == 15.0

    s3 = s1 + s2
    assert len(s3) == 2
    assert s3[0] == 17.0

    s3 = s1 - s2
    assert len(s3) == 2
    assert s3[0] == -13.0

    s3 = s1 * s2
    assert len(s3) == 2
    assert s3[0] == 30.0

    s3 = s2 / s1
    assert len(s3) == 2
    assert round(s3[0], 3) == 7.5

    s3 = (s1 + timedelta(days=1)) + s2
    assert len(s3) == 3
    assert s3[0] == 16.0

    s3 = s2 / 6.0
    assert len(s3) == 3
    assert round(s3[2], 3) == 3.0
