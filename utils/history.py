import pandas as pd

class History:
    def __init__(self):
        self._records = []

    def log(self, **kwargs):
        """
        Log a single step or epoch of training data.
        Example: log(epoch=1, loss=0.5, accuracy=0.8)
        """
        self._records.append(kwargs)

    @property
    def df(self):
        """
        Returns the full training history as a pandas DataFrame.
        """
        return pd.DataFrame(self._records)

    def latest(self):
        """
        Return the most recent log entry as a pandas Series.
        """
        return self.df.iloc[-1] if not self.df.empty else None

    def get(self, key):
        """
        Return a Series of values for a specific metric.
        """
        return self.df[key] if key in self.df else None

    def __str__(self):
        return str(self.latest())