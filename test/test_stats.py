import pytest

from covidsimulation.stats import Stats


def test_get_metric():
    stats = Stats.load('test/fixtures/teste.pkl')
    assert len(stats.get_metric('deaths').mean) == 21
    assert stats.get_metric('pc_infected', ['classe_abc+', 'classe_e'], ['80+', '70-79']).high[0]
    assert stats.get_metric('pc_infected', ['classe_abc+', 'classe_e'], ['80+', '70-79'], daily=True).high[0]
    assert stats.get_metric('deaths', ['classe_abc+', 'classe_e'], '80+').low[0] == 0.0
    assert stats.get_metric('deaths', 'classe_abc+', '80+').low[0] == 0.0
