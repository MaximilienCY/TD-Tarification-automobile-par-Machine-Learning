"""Combine the preserved severity estimator and the imported frequency recipe."""
import numpy as np
import pandas as pd
from .frequency import RAW_COLUMNS


class GlobalModel:
    def __init__(self, frequency, severity):
        self.frequency = frequency
        self.severity = severity

    def predict_components(self, frame):
        missing = sorted((set(RAW_COLUMNS) | {'index'}) - set(frame.columns))
        if missing:
            raise ValueError('Missing input columns: ' + ', '.join(missing))
        if frame['index'].isna().any() or not frame['index'].is_unique:
            raise ValueError('index must be present and unique.')
        # Both estimators receive exactly the same ordered contracts.
        p = np.asarray(self.frequency.predict_proba(frame), dtype=float)
        g = np.asarray(self.severity.predict(frame[RAW_COLUMNS]), dtype=float)
        if p.shape != (len(frame),) or g.shape != p.shape:
            raise ValueError('Component prediction lengths do not match the input.')
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Invalid claim probabilities.')
        if not np.isfinite(g).all() or not (g > 0).all():
            raise ValueError('Invalid conditional severity.')
        return pd.DataFrame({'index': frame['index'].to_numpy(), 'probabilite_sinistre': p,
                             'gravite_predite': g, 'prime_predite': p * g})

    def predict(self, frame):
        return self.predict_components(frame)['prime_predite'].to_numpy()
