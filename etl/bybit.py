"""OI/Funding загрузка из локальных файлов (API можно добавить позже)."""
import pandas as pd

def load_bybit_derivs(oi_path: str, funding_path: str) -> pd.DataFrame:
    oi = pd.read_parquet(oi_path) if oi_path.endswith('.parquet') else pd.read_csv(oi_path)
    fd = pd.read_parquet(funding_path) if funding_path.endswith('.parquet') else pd.read_csv(funding_path)
    for d in (oi, fd):
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
        d.set_index('timestamp', inplace=True)
    oi_h = oi.resample('1H').last().rename(columns={'open_interest':'oi'})
    fd_h = fd.resample('1H').last().rename(columns={'funding_rate':'funding'})
    return oi_h.join(fd_h, how='outer').sort_index()
