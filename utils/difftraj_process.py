import pandas as pd
import numpy as np
import os
import datetime
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from geographiclib.geodesic import Geodesic

CONFIG = {
    "traj_length": 200,  # 与config.data.traj_length一致
    "grid_size": 16,     # 16x16网格
    "input_csv": r"D:\BaiduNetdiskDownload\2016年成都滴滴轨迹数据\filter.csv",
    "output_dir": "./processed_data",
    "min_points": 120    
}


def haversine_distance(lng1, lat1, lng2, lat2):
    try:
        return Geodesic.WGS84.Inverse(lat1, lng1, lat2, lng2)['s12']
    except:
        R = 6371000  
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)** 2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def timestamp_to_depature_index(dt):
    total_minutes = dt.hour * 60 + dt.minute  
    return total_minutes // 5 


print("="*50)
print(f"目标格式：traj[N, 2, {CONFIG['traj_length']}], head[N, 8]")
print("="*50)


os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("\n[1/5] 读取源数据...")
read_csv_args = {
    "names": ['司机ID', '订单ID', 'GPS时间', '经度', '纬度'],
    "dtype": {
        '司机ID': str,
        '订单ID': str,
        '经度': str,
        '纬度': str,
        'GPS时间': str
    },
    "chunksize": 100000,
    "encoding": "utf-8"
}

if pd.__version__ >= "1.3.0":
    read_csv_args["on_bad_lines"] = "skip"
else:
    read_csv_args["error_bad_lines"] = False

chunks = []
for chunk in tqdm(
    pd.read_csv(CONFIG["input_csv"], **read_csv_args),
    desc="读取CSV文件"
):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"原始数据共 {len(df):,} 行")

df['GPS时间'] = pd.to_datetime(df['GPS时间'], errors='coerce')  # 转换为datetime
df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
df = df.dropna(subset=['订单ID', 'GPS时间', '经度', '纬度'])
df = df[(df['经度'] > 0) & (df['纬度'] > 0)]
print(f"清洗后保留 {len(df):,} 行")


print("\n[2/5] 按订单分组...")
grouped = df.groupby('订单ID')
valid_orders = [oid for oid, group in grouped if len(group) >= CONFIG["min_points"]]
df = df[df['订单ID'].isin(valid_orders)]
grouped = df.groupby('订单ID')
print(f"有效订单数量：{len(grouped)}（每条轨迹≥{CONFIG['min_points']}个点）")

if len(grouped) == 0:
    raise ValueError(f"无有效订单！请降低min_points（当前{CONFIG['min_points']}）")


print("\n[3/5] 计算地理范围...")
all_lng = np.concatenate([group['经度'].values for _, group in grouped])
all_lat = np.concatenate([group['纬度'].values for _, group in grouped])
lng_min, lng_max = np.min(all_lng), np.max(all_lng)
lat_min, lat_max = np.min(all_lat), np.max(all_lat)
print(f"经度范围：[{lng_min:.6f}, {lng_max:.6f}]，纬度范围：[{lat_min:.6f}, {lat_max:.6f}]")


print("\n[4/5] 生成traj.npy...")
traj_list = []
for oid, group in tqdm(grouped, desc="处理轨迹坐标"):
    group = group.sort_values('GPS时间')
    lat = group['纬度'].values.astype(np.float64)
    lng = group['经度'].values.astype(np.float64)
    n_points = len(group)
    
    old_indices = np.arange(n_points)
    new_indices = np.linspace(0, n_points-1, CONFIG["traj_length"])
    interp_lat = interp1d(old_indices, lat, kind='linear', assume_sorted=True)(new_indices)
    interp_lng = interp1d(old_indices, lng, kind='linear', assume_sorted=True)(new_indices)
    
    traj = np.stack([interp_lat, interp_lng], axis=0).astype(np.float32)
    traj_list.append(traj)

traj_array = np.stack(traj_list, axis=0)
mean_lat, mean_lng = np.mean(traj_array, axis=(0, 2))
std_lat, std_lng = np.std(traj_array, axis=(0, 2)) + 1e-8
traj_array[:, 0, :] = (traj_array[:, 0, :] - mean_lat) / std_lat
traj_array[:, 1, :] = (traj_array[:, 1, :] - mean_lng) / std_lng

assert traj_array.shape == (len(grouped), 2, CONFIG["traj_length"]), \
    f"traj形状错误！实际{traj_array.shape}，预期[{len(grouped)}, 2, {CONFIG['traj_length']}]"
np.save(os.path.join(CONFIG["output_dir"], "traj.npy"), traj_array)
print(f"traj.npy 保存完成，形状：{traj_array.shape}")


print("\n[5/5] 生成head.npy...")
def get_grid_id(lng, lat):
    grid_col = int(((lng - lng_min) / (lng_max - lng_min + 1e-8)) * (CONFIG["grid_size"] - 1))
    grid_row = int(((lat - lat_min) / (lat_max - lat_min + 1e-8)) * (CONFIG["grid_size"] - 1))
    return np.clip(grid_row * CONFIG["grid_size"] + grid_col, 0, 255)

head_features = []
for oid, group in tqdm(grouped, desc="提取条件特征"):
    group = group.sort_values('GPS时间')
    coords = group[['经度', '纬度']].values
    start_time = group['GPS时间'].iloc[0]  # 出发时间（datetime类型）
    
    depature_idx = timestamp_to_depature_index(start_time)
    depature_idx = np.clip(depature_idx, 0, 287)  # 强制范围

    total_distance = 0.0
    for i in range(1, len(coords)):
        total_distance += haversine_distance(
            coords[i-1][0], coords[i-1][1],
            coords[i][0], coords[i][1]
        )
    

    duration = (group['GPS时间'].iloc[-1] - start_time).total_seconds()
    duration = max(duration, 1e-8)
    
    raw_length = len(group)

    avg_step = total_distance / (raw_length - 1) if raw_length > 1 else 0.0
    
    avg_speed = total_distance / duration
    
    sid = get_grid_id(coords[0][0], coords[0][1])
    
    eid = get_grid_id(coords[-1][0], coords[-1][1])
    
    head_features.append([
        depature_idx,     
        total_distance,  
        duration,         
        raw_length,      
        avg_step,         
        avg_speed,        
        sid,              
        eid               
    ])


head_array = np.array(head_features, dtype=np.float32)
scaler = StandardScaler()
head_array[:, 1:6] = scaler.fit_transform(head_array[:, 1:6])  # 只标准化1~5列

head_array[:, 0] = head_array[:, 0].astype(int)
head_array[:, 6] = head_array[:, 6].astype(int)
head_array[:, 7] = head_array[:, 7].astype(int)

depature_range = (head_array[:, 0].min(), head_array[:, 0].max())
sid_range = (head_array[:, 6].min(), head_array[:, 6].max())
eid_range = (head_array[:, 7].min(), head_array[:, 7].max())
print(f"depature范围: {depature_range}（应在0~287）")
print(f"sid范围: {sid_range}（应在0~255）")
print(f"eid范围: {eid_range}（应在0~255）")

np.save(os.path.join(CONFIG["output_dir"], "head.npy"), head_array)
print(f"head.npy 保存完成，形状：{head_array.shape}")


norm_params = {
    "traj": {
        "mean_lat": mean_lat, "std_lat": std_lat,
        "mean_lng": mean_lng, "std_lng": std_lng
    },
    "head": {
        "mean": scaler.mean_, "std": scaler.scale_
    }
}
np.save(os.path.join(CONFIG["output_dir"], "norm_params.npy"), norm_params)

print("\n" + "="*50)
print("数据转换完成  输出文件：")
print(f"traj.npy: {os.path.abspath(os.path.join(CONFIG['output_dir'], 'traj.npy'))}")
print(f"head.npy: {os.path.abspath(os.path.join(CONFIG['output_dir'], 'head.npy'))}")
print("="*50)
