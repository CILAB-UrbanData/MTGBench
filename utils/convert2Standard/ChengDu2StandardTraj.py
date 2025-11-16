#对成都适配
import pandas as pd
import geopandas as gpd
import os
import glob

def addCol2shp(csv_path = r"/path/to/edge_mapping.csv",shp_path = r"/path/to/road_network.shp",out_path = r"/path/to/road_network_with_edgeid.shp" ):
    '''
    把RoadMatch/processed/roads_chengdu_edges.csv中的edge_id按照shp中的osm_id（后面结果都是跟着edge_id来的）加入到相应的路网shp文件中
    '''
    # 1. 读入 edge 映射 csv
           # 你的 csv
    df = pd.read_csv(csv_path)

    # 如果 osm_id 在 shp 里是字符串类型，可以统一一下类型
    df["osm_id"] = df["osm_id"].astype(str)

    # 只保留需要 join 的字段（可选）
    df = df[["osm_id", "edge_id"]]

    # 2. 读入原始路网 shp
          # 原始 shp
    gdf = gpd.read_file(shp_path)

    # 如果 shp 里的 osm_id 类型不同，也统一一下
    gdf["osm_id"] = gdf["osm_id"].astype(str)

    # 3. 按 osm_id 做属性表合并，把 edge_id 加进去
    gdf_merged = gdf.merge(df, on="osm_id", how="left")

    # 4. 保存新的 shp（或保存为 gpkg/geojson 也可以）
    
    gdf_merged.to_file(out_path, encoding="utf-8")

    print("完成！新文件已写入：", out_path)

def convertTraj2SF(TRJ_COL = "traj_id",
    DRIVER_COL_CANDIDATES = ["driver_id", "司机ID"],
    TS_COL_CANDIDATES = ["timestamp", "GPS时间"],
    EDGE_COL = "edge_id",
    DWELL_COL = "dwell_time",    
    input_dir = "data/GaiyaData/RoadMatch/processed", 
    pattern = "matched_*.csv",              
    output_csv = "data/GaiyaData/TRACK/traj_converted.csv"):
    """
    将原始轨迹 csv 转为目标格式：
    driver_id, traj_id, offsets, segment_sequence, gps_points

    需求实现：
    1. traj_id 为空时，用 driver_id 填充
    2. offsets 不关心内容，用两个空元组占位：[(), ()]
    3. 原始时间为北京时间，转换成 Unix time（秒）
    4. segment_sequence 中 average speed 无信息，用 None 占位
    5. 多个 csv 处理后合并输出到一个 csv 文件
    """

    def find_first_exist(df, candidates):
        """在给定候选列名中找到第一个存在于 df 的列名"""
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"在列 {df.columns.tolist()} 中找不到 {candidates}")


    def to_unix_from_beijing(ts_series: pd.Series) -> pd.Series:
        """
        将北京时间字符串转为 Unix 时间戳（秒）
        """
        dt = pd.to_datetime(ts_series, errors="coerce")
        # 视为北京时间
        dt = dt.dt.tz_localize("Asia/Shanghai", nonexistent="shift_forward")
        return (dt.view("int64") // 10**9).astype("float64")


    def build_segments_for_traj(grp: pd.DataFrame,
                                edge_col: str,
                                unix_col: str,
                                dwell_col: str):
        """
        对单个 traj 的数据构造 segment_sequence:
        [edge_id, enter_time_unix, None, duration_seconds]

        将连续相同 edge_id 合并，duration = 5 * sum(dwell_time)
        """
        grp = grp.sort_values(unix_col)
        segments = []

        prev_edge = None
        start_unix = None
        dwell_sum = 0.0

        for edge, unix_t, dwell in zip(grp[edge_col],
                                    grp[unix_col],
                                    grp[dwell_col].fillna(0)):
            if pd.isna(edge) or pd.isna(unix_t):
                continue

            if prev_edge is None:  # 第一条
                prev_edge = edge
                start_unix = unix_t
                dwell_sum = float(dwell)
            elif edge == prev_edge:
                dwell_sum += float(dwell)
            else:
                # 结束上一段
                duration = dwell_sum * 5.0
                segments.append([int(prev_edge), float(start_unix), None, duration])

                # 开始新一段
                prev_edge = edge
                start_unix = unix_t
                dwell_sum = float(dwell)

        # 收尾
        if prev_edge is not None:
            duration = dwell_sum * 5.0
            segments.append([int(prev_edge), float(start_unix), None, duration])

        return segments


    def process_one_file(path, out_path, write_header):
        print(f"处理文件: {path}")
        df = pd.read_csv(path)

        driver_col = find_first_exist(df, DRIVER_COL_CANDIDATES)
        ts_col = find_first_exist(df, TS_COL_CANDIDATES)

        # 1. traj_id 为空，用 driver_id 填充
        if TRJ_COL not in df.columns:
            raise ValueError(f"找不到列 {TRJ_COL}")
        df[TRJ_COL] = df[TRJ_COL].fillna(df[driver_col]).astype(str)

        # 3. 时间（北京时间）转 Unix
        df["unix_time"] = to_unix_from_beijing(df[ts_col])

        # 确保 dwell_time 存在
        if DWELL_COL not in df.columns:
            raise ValueError(f"找不到列 {DWELL_COL}")

        # 只保留需要的列减小内存
        needed_cols = [TRJ_COL, driver_col, EDGE_COL, "unix_time", DWELL_COL]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少列: {missing}")
        df = df[needed_cols]

        # 按 traj_id 分组生成输出
        rows = []
        for traj_id, grp in df.groupby(TRJ_COL):
            driver_id = str(grp[driver_col].iloc[0])

            # 2. offsets 用两个空元组占位
            offsets = [(), ()]

            # 4. segment_sequence：average speed 为 None
            segments = build_segments_for_traj(
                grp,
                edge_col=EDGE_COL,
                unix_col="unix_time",
                dwell_col=DWELL_COL,
            )

            rows.append(
                {
                    "driver_id": driver_id,
                    "traj_id": traj_id,
                    "offsets": str(offsets),
                    "segment_sequence": str(segments),
                }
            )

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, mode="a", index=False, header=write_header, encoding="utf-8")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print("没有找到输入文件")
        return

    first = True
    for f in files:
        process_one_file(f, output_csv, write_header=first)
        first = False

    print("全部完成，输出文件：", output_csv)


if __name__ == "__main__":
    addCol2shp(
        csv_path = r"data/GaiyaData/RoadMatch/processed/roads_chengdu_edges.csv",
        shp_path = r"data/GaiyaData/RoadMatch/roads_chengdu.shp",
        out_path = r"data/GaiyaData/TRACK/roads_chengdu.shp"
    )
    convertTraj2SF(
        input_dir = "data/GaiyaData/RoadMatch/processed", 
        pattern = "matched_*.csv",              
        output_csv = "data/GaiyaData/TRACK/traj_converted.csv"
    )