import numpy as np
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
def minmax_norm(series):
    s = series.astype(float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(hi - lo) or (hi - lo) == 0:
        return np.zeros_like(s, dtype=float)
    return (s - lo) / (hi - lo)
