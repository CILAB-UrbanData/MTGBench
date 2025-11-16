import os
import geopandas as gpd
import pickle

def build_full_segment_vocab_from_trajs_and_edge(raw_map, roadid_col=None, out_vocab_file=None):
    """
    从 raw_map 文件构造 segment vocabulary（road_id -> idx）。
    """
    segs = set()
    if os.path.exists(raw_map):
        cols_to_keep = [roadid_col]
        cols_to_copy = [roadid_col]
        new_col_name = "weight"
        speed = 583  # m/min ~ 35km/h
        gdf = gpd.read_file(raw_map)
        missing_cols = [c for c in cols_to_keep if c not in gdf.columns]  
        if missing_cols:
            raise ValueError(f"以下字段在源文件中不存在: {missing_cols}")
        new_gdf = gdf[cols_to_copy].copy()
        new_gdf.rename(columns={roadid_col: "road_id"}, inplace=True)
        # 可选：travel_time = gdf["length"] / speed  # 未实际使用
        for rid in new_gdf["road_id"]:
            try:
                segs.add(int(rid))
            except:
                continue
    else:
        raise ValueError(f"raw_map file {raw_map} not found")

    segs = sorted(list(segs))
    seg2idx = {int(s): i for i, s in enumerate(segs)}
    idx2seg = segs
    if out_vocab_file:
        with open(out_vocab_file, 'wb') as fh:
            pickle.dump({'seg2idx': seg2idx, 'idx2seg': idx2seg}, fh)
    return seg2idx, idx2seg