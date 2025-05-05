import numpy as np
import matplotlib.pyplot as plt

# 参数设置
z = 2.0  # 所有点的z坐标
speaker_spacing = 0.1  # 扬声器间距(米)
num_speakers = 8       # 扬声器数量
array_center = (0, 0, z)  # 扬声器阵列中心

bright_angle = np.deg2rad(-30)  # 亮区角度(弧度)
dark_angle = np.deg2rad(30)     # 暗区角度(弧度)
zone_distance = 1.5             # 区中心距离(米)
control_point_spacing = 0.04    # 控制点间距(米)
num_control_points = 15         # 每个区的控制点数

# 生成扬声器坐标(沿x轴对称分布)
speaker_coords = []
start_x = -(num_speakers-1)*speaker_spacing/2
for i in range(num_speakers):
    x = start_x + i*speaker_spacing
    speaker_coords.append((x, 0, z))

# 生成控制点坐标函数
def generate_zone_points(angle, num_points, center_distance):
    points = []
    center_x = center_distance * np.cos(angle)
    center_y = center_distance * np.sin(angle)
    
    # 生成从-7到7的索引(共15个点)
    for i in range(-(num_points//2), num_points//2 + 1):
        dx = i * control_point_spacing * np.cos(angle)
        dy = i * control_point_spacing * np.sin(angle)
        points.append((
            center_x + dx,
            center_y + dy,
            z
        ))
    return points

# 生成亮区和暗区控制点
bright_zone = generate_zone_points(bright_angle, num_control_points, zone_distance)
dark_zone = generate_zone_points(dark_angle, num_control_points, zone_distance)

# 打印结果
print("扬声器坐标：")
for i, coord in enumerate(speaker_coords):
    print(f"Speaker {i+1}: {coord}")

print("\n亮区控制点坐标：")
for i, coord in enumerate(bright_zone):
    print(f"Bright Point {i+1}: {coord}")

print("\n暗区控制点坐标：")
for i, coord in enumerate(dark_zone):
    print(f"Dark Point {i+1}: {coord}")

dark_zone = np.array(dark_zone)
bright_zone = np.array(bright_zone)
speaker_coords = np.array(speaker_coords)

