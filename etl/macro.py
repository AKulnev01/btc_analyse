"""Флаги окон вокруг FOMC/CPI/NFP: макро-волатильность."""
import pandas as pd

def make_macro_windows(events: pd.DataFrame, start='-60min', end='+120min') -> pd.DataFrame:
    events = events.copy()
    events['ts'] = pd.to_datetime(events['ts'], utc=True)
    idx = pd.date_range(events['ts'].min().floor('H')-pd.Timedelta('2D'),
                        events['ts'].max().ceil('H')+pd.Timedelta('2D'),
                        freq='1H', tz='UTC')
    out = pd.DataFrame(index=idx)
    for ev in events['event'].unique():
        flag = pd.Series(0, index=idx)
        for ts in events.loc[events['event']==ev, 'ts']:
            s = (ts + pd.Timedelta(start)).floor('H')
            e = (ts + pd.Timedelta(end)).ceil('H')
            flag.loc[s:e] = 1
        out[f'macro_{ev.lower()}_win'] = flag.astype('int8')
    return out
