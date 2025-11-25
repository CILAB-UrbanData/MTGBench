import geopandas as gpd
import numpy as np
import torch
from collections import defaultdict

def build_full_edges_from_shp_using_vocab(shp_path,
                                          seg2idx,
                                          roadid_col='fid',
                                          from_col='u',
                                          to_col='v',
                                          geom_col='geometry',
                                          assume_uv_present=True,
                                          coord_round=6,
                                          verbose=True,
                                          skip_missing=True):
    """
    使用已有 seg2idx 从 shapefile 构建 full_edge_index（segment-level adjacency）。
    """
    gdf = gpd.read_file(shp_path)
    if verbose:
        print(f"[shp->full_edges] load {len(gdf)} features from {shp_path}")

    if roadid_col not in gdf.columns:
        raise ValueError(f"roadid_col '{roadid_col}' not found in shapefile columns: {gdf.columns.tolist()}")

    seg_to_endpoints = {}   # seg_idx -> (start_key, end_key)
    missing_roadids = set()
    coord_map = {}
    next_coord_node = 0

    use_uv = assume_uv_present and (from_col in gdf.columns) and (to_col in gdf.columns)
    if use_uv and verbose:
        print("[shp->full_edges] using explicit from/to columns for endpoints")

    for i, row in gdf.iterrows():
        raw_rid = row[roadid_col]
        try:
            rid_key = int(raw_rid)
        except Exception:
            rid_key = raw_rid

        if rid_key not in seg2idx:
            missing_roadids.add(rid_key)
            if skip_missing:
                continue
            else:
                raise KeyError(f"roadid {rid_key} from shapefile not found in provided seg2idx vocab")

        seg_idx = seg2idx[rid_key]

        if use_uv:
            u_raw = row[from_col]; v_raw = row[to_col]
            try:
                u_key = int(u_raw)
            except:
                u_key = u_raw
            try:
                v_key = int(v_raw)
            except:
                v_key = v_raw
            seg_to_endpoints[seg_idx] = (("nid", u_key), ("nid", v_key))
        else:
            geom = row[geom_col]
            if geom is None:
                if verbose:
                    print(f"[shp->full_edges] warning: seg {rid_key} has no geometry, skipping")
                continue
            try:
                coords = list(geom.coords)
            except Exception:
                if geom.geom_type == 'MultiLineString':
                    coords = list(list(geom)[0].coords)
                else:
                    raise
            start = coords[0]; end = coords[-1]
            start_k = (round(float(start[0]), coord_round), round(float(start[1]), coord_round))
            end_k = (round(float(end[0]), coord_round), round(float(end[1]), coord_round))
            if start_k not in coord_map:
                coord_map[start_k] = f"c{next_coord_node}"; next_coord_node += 1
            if end_k not in coord_map:
                coord_map[end_k] = f"c{next_coord_node}"; next_coord_node += 1
            seg_to_endpoints[seg_idx] = (("coord", coord_map[start_k]), ("coord", coord_map[end_k]))

    if verbose:
        print(f"[shp->full_edges] found {len(seg_to_endpoints)} segments matched to vocab; {len(missing_roadids)} missing roadids")

    start_to_segs = defaultdict(list)
    end_of_seg = {}
    for seg_idx, (start_key, end_key) in seg_to_endpoints.items():
        start_to_segs[start_key].append(seg_idx)
        end_of_seg[seg_idx] = end_key

    src_list, dst_list = [], []
    for A, end_key in end_of_seg.items():
        b_list = start_to_segs.get(end_key, [])
        for B in b_list:
            src_list.append(A)
            dst_list.append(B)

    if len(src_list) == 0:
        fe = np.zeros((2, 0), dtype=np.int64)
    else:
        fe = np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=0)

    info = {
        'num_features_loaded': len(gdf),
        'num_segments_matched': len(seg_to_endpoints),
        'num_missing_roadids': len(missing_roadids),
        'missing_roadids': list(missing_roadids),
        'num_edges': fe.shape[1]
    }
    if verbose:
        print(f"[shp->full_edges] built full_edge_index with {fe.shape[1]} edges")

    return torch.LongTensor(fe), info