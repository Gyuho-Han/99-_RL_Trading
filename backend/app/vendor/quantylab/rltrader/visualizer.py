"""No-op Visualizer.

The original quantylab/rltrader Visualizer renders matplotlib/mplfinance PNGs
per epoch. In this web prototype we render charts in the React frontend instead,
so the heavy plotting dependency is removed and every method is a no-op. The
class keeps the exact interface the learners call (prepare/clear/plot/save) so
the rest of the framework runs unchanged.
"""


class Visualizer:
    def __init__(self, vnet=False):
        self.vnet = vnet

    def prepare(self, chart_data=None, title=None):
        return None

    def clear(self, xlim=None):
        return None

    def plot(self, *args, **kwargs):
        return None

    def save(self, path=None):
        return None
