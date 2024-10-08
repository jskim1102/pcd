1002 : 3d pcd 후보 영역 지정후 crop(①)하고 a,b,c,d 좌표로 pcd 추출(②)

1003 : pcd xy평면 회전(③) 및 노이즈 제거 후 3d bbox fitting(④)

1004 : filtering된 pcd(①+②+③+④) 저장

1008 : 노이즈 제거 코드 수정 및 계층 분리된 height map 저장(⑤)


# Origin Code [nerf_1004]
## ① ply 파일 읽어서 3d pcd의 후보 영역을 지정하기(bb내부 영역에 있는 pcd만 추출 : pcd_crop)
```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA

path = 'N:/2024/[1]_kjs_lee/[3]_pcd/0930_js_004.ply'
pcd = o3d.io.read_point_cloud(path)

bb_crop = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -.5), max_bound=(1, 1, .5))
pcd_crop = pcd.crop(bb_crop)
bb_crop.color = (1, 0, 0) 

# 3D 시각화
o3d.visualization.draw_geometries([pcd, bb_crop])
o3d.visualization.draw_geometries([pcd_crop, bb_crop])
```


## ② 3d pcd를 수직 위에서 바라본 2차원 BEV 이미지로 시각화하기 -> x, y 좌표값을 확인할수 있음(확인한 좌표값에 따라 철근 구조물에 해당하는 pcd를 추출하기 위한 폴리곤 좌표 A, B, C, D를 설정) -> A, B, C, D 좌표값의 폴리곤 내부의 pcd만 추출
```python
xyz, rgb = np.asarray(pcd_crop.points), np.asarray(pcd_crop.colors)
x = xyz[:, 0]
y = xyz[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, c='blue')  # 포인트 클라우드 시각화(bb)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Cropped Point Cloud')
plt.grid(True)
plt.axis('equal')
plt.show()

# A, B, C, D 점 정의 및 시각화
A = (-0.75, 0.75)
B = (-0.48, -0.48)
C = (0.47, -0.23)
D = (0.18, 0.95)

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, c='blue')  # 포인트 클라우드
plt.scatter(*zip(A, B, C, D), s=100, c='red')  # A, B, C, D 점 표시
plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Filtered Point Cloud (with A, B, C, D Points)')
plt.grid(True)
plt.axis('equal')
plt.show()

polygon = np.array([A, B, C, D])
path_ = Path(polygon)

xy_points = xyz[:, :2]  # X와 Y 좌표만 사용
mask = path_.contains_points(xy_points)

xyz_mask, rgb_mask = xyz[mask], rgb[mask]
pcd_mask = o3d.geometry.PointCloud()
pcd_mask.points = o3d.utility.Vector3dVector(xyz_mask)
pcd_mask.colors = o3d.utility.Vector3dVector(rgb_mask)

o3d.visualization.draw_geometries([pcd_mask, bb_crop])

x_mask = xyz_mask[:, 0]
y_mask = xyz_mask[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x_mask, y_mask, s=1, c='blue') 
plt.scatter(*zip(A, B, C, D), s=100, c='red')
plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Filtered Point Cloud (with A, B, C, D Points)')
plt.grid(True)
plt.axis('equal')
plt.show()
```

## ③ 철근 pcd xy평면 회전
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(pcd_mask.points)


matrix_rot = pca.components_.T
xyz_rot = pcd_mask.points @ matrix_rot

pcd_rot = o3d.geometry.PointCloud()
pcd_rot.points = o3d.utility.Vector3dVector(xyz_rot)
pcd_rot.colors = pcd_mask.colors


bb_rot = pcd_rot.get_axis_aligned_bounding_box()
bb_rot.color = (1, 0, 0)

print(bb_rot.min_bound, bb_rot.max_bound)

o3d.visualization.draw_geometries([pcd_rot, bb_crop])
o3d.visualization.draw_geometries([pcd_rot, bb_rot])


x_rot = xyz_rot[:, 0]
y_rot = xyz_rot[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x_rot, y_rot, s=1, c='blue')
plt.scatter(*zip(A, B, C, D), s=100, c='red')
plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Aligned Point Cloud (with A, B, C, D Points)')
plt.grid(True)
plt.axis('equal')
plt.show()
```

## ④ xy평면의 point 개수가 임계값 이하인 z층 point를 삭제하여 noise 제거 -> 3차원 경계상자 fitting
```python
pcd_rot_np = np.copy(np.asarray(pcd_rot.points))
pcd_rot_colors = np.asarray(pcd_rot.colors)  # 원본 색상 데이터도 가져오기
z_values = np.unique(np.round(pcd_rot_np[:, 2], decimals=4))

filtered_points = []
filtered_colors = []  # 필터링된 색상을 저장할 리스트

threshold = 20

for z in z_values:
    # 해당 z 값을 기준으로 xy 평면에 있는 포인트 선택
    mask = np.abs(pcd_rot_np[:, 2] - z) < 1e-4
    points_in_slice = pcd_rot_np[mask]
    colors_in_slice = pcd_rot_colors[mask]  # 해당 포인트에 맞는 색상도 가져오기
    
    # 임계값 이상이면 해당 슬라이스의 포인트와 색상을 유지
    if len(points_in_slice) >= threshold:
        filtered_points.append(points_in_slice)
        filtered_colors.append(colors_in_slice)

# 필터링된 포인트와 색상 배열로 변환
filtered_point_cloud = np.vstack(filtered_points)
filtered_colors_cloud = np.vstack(filtered_colors)

# Open3D 포인트 클라우드 객체 생성 (필터링된 데이터)
pcd_fit = o3d.geometry.PointCloud()
pcd_fit.points = o3d.utility.Vector3dVector(filtered_point_cloud)
pcd_fit.colors = o3d.utility.Vector3dVector(filtered_colors_cloud)

o3d.visualization.draw_geometries([pcd_fit, bb_rot])

z_values = np.asarray(pcd_fit.points)[:, 2]
z_min, z_max = z_values.min(), z_values.max()

min_bound, max_bound = bb_rot.min_bound.copy(), bb_rot.max_bound.copy()
min_bound[2], max_bound[2] = z_min, z_max

bb_fit = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
bb_fit.color = (1, 0, 0)

o3d.visualization.draw_geometries([pcd_fit, bb_fit])

directory = path.split('[3]_pcd')[0]
filename = path.split('/')[-1].split('.')[0]

output_dir = directory + '[4]_filtered_pcd/'
output_path = output_dir + filename + '_filtered.ply'

# # 저장 및 시각화
# o3d.io.write_point_cloud(output_path,pcd_fit)

# pcd_filtered = o3d.io.read_point_cloud(output_path)
# o3d.visualization.draw_geometries([pcd_filtered])
```

## ⑤ Z값을 기준으로 계층 분리 후 Height Map 출력 및 저장
```python
import os

# 포인트 클라우드를 numpy 배열로 변환합니다.
fit_xyz = np.asarray(pcd_fit.points)
fit_rgb = np.asarray(pcd_fit.colors)

# --- Z 값 기반으로 1층과 2층 분리 --- 
z_vals = fit_xyz[:, 2]  # Z 값 추출
z_min = z_vals.min()
z_max = z_vals.max()

# Z 값에 기반해 1층과 2층을 분리
threshold = z_max + (z_min + z_max) * 0.08 

# 1층 철근 필터링
layer_1_mask = z_vals <= threshold
layer_1_xyz = fit_xyz[layer_1_mask]
layer_1_rgb = fit_rgb[layer_1_mask]

# 2층 철근 필터링
layer_2_mask = z_vals > threshold
layer_2_xyz = fit_xyz[layer_2_mask]
layer_2_rgb = fit_rgb[layer_2_mask]

# 1층 PointCloud 생성
layer_1_pcd = o3d.geometry.PointCloud()
layer_1_pcd.points = o3d.utility.Vector3dVector(layer_1_xyz)
layer_1_pcd.colors = o3d.utility.Vector3dVector(layer_1_rgb)

o3d.visualization.draw_geometries([layer_1_pcd])

# 2층 PointCloud 생성
layer_2_pcd = o3d.geometry.PointCloud()
layer_2_pcd.points = o3d.utility.Vector3dVector(layer_2_xyz)
layer_2_pcd.colors = o3d.utility.Vector3dVector(layer_2_rgb)

o3d.visualization.draw_geometries([layer_2_pcd])

# Height Map을 생성하는 함수 (Z 하한 및 상한 추가)
def generate_height_map(points, grid_size=0.001, z_lower_limit=None, z_upper_limit=None):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    # Z 하한 및 상한 설정 (필요시)
    if z_lower_limit is not None:
        points = points[points[:, 2] >= z_lower_limit]
    if z_upper_limit is not None:
        points = points[points[:, 2] <= z_upper_limit]

    # x, y 축의 그리드를 만듭니다.
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)

    # 높이 맵을 초기화합니다.
    height_map = np.full((len(y_grid), len(x_grid)), np.nan)

    # 포인트들을 XY grid에 맵핑하여 Z 값을 height map으로 변환합니다.
    for point in points:
        x_idx = int((point[0] - x_min) / grid_size)
        y_idx = int((point[1] - y_min) / grid_size)
        if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
            height_map[y_idx, x_idx] = point[2]

    # NaN 값을 최소 높이 값으로 대체합니다.
    min_height = np.nanmin(height_map)
    height_map = np.nan_to_num(height_map, nan=min_height)

    return height_map, x_min, x_max, y_min, y_max


# 1층과 2층 각각에 대해 Height Map을 생성합니다.
layer_1_height_map, x_min_1, x_max_1, y_min_1, y_max_1 = generate_height_map(layer_1_xyz, z_lower_limit=None)
layer_2_height_map, x_min_2, x_max_2, y_min_2, y_max_2 = generate_height_map(layer_2_xyz)

layer_path = directory + '[5]_height_map/'
if not os.path.exists(layer_path): os.makedirs(layer_path)


plt.imshow(layer_1_height_map, cmap='viridis', extent=[x_min_1, x_max_1, y_min_1, y_max_1])
plt.axis('off')  # 축을 제거합니다.
# plt.show()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 여백을 제거합니다.
plt.imsave(layer_path + filename + '_layer_1.png', layer_1_height_map, cmap='viridis')


plt.imshow(layer_2_height_map, cmap='viridis', extent=[x_min_2, x_max_2, y_min_2, y_max_2])
plt.axis('off')  # 축을 제거합니다.
# plt.show()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 여백을 제거합니다.
plt.imsave(layer_path + filename + '_layer_2.png', layer_2_height_map, cmap='viridis')
```
