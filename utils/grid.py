def calculate_grid_bounds(all_lng: np.ndarray, all_lat: np.ndarray) -> tuple:
    return np.min(all_lng), np.max(all_lng), np.min(all_lat), np.max(all_lat)


def batch_get_grid_ids(
    all_lng: np.ndarray, 
    all_lat: np.ndarray, 
    grid_bounds: tuple, 
    grid_size: int = 16
) -> np.ndarray:
    lng_min, lng_max, lat_min, lat_max = grid_bounds
    grid_col = np.round(((all_lng - lng_min) / (lng_max - lng_min)) * (grid_size - 1)).astype(int)
    grid_row = np.round(((all_lat - lat_min) / (lat_max - lat_min)) * (grid_size - 1)).astype(int)
    return grid_row * grid_size + grid_col
