1002 : 3d pcd 영역 지정후 crop하고 a,b,c,d 좌표로 pcd 추출  
1003 : pcd xy평면 회전 및 노이즈 제거 후 3d bbox fitting  
1004 : filtering된 pcd 저장  
1008 : 노이즈 제거 코드 수정 및 계층 분리된 height map 저장

# Origin Code [nerf_1004]
## 1. ply 파일 읽어서 3d pcd의 영역을 지정하기(bb내부 영역에 있는 pcd만 추출 : pcd_crop)
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


## 2. 3d pcd를 수직 위에서 바라본 2차원 BEV 이미지로 시각화하기 -> x, y 좌표값을 확인할수 있음(확인한 좌표값에 따라 철근 구조물에 해당하는 pcd를 추출하기 위한 폴리곤 좌표 A, B, C, D를 설정)
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
```

## 3. A, B, C, D 좌표값의 폴리곤 내부의 pcd만 추출
```python
polygon = np.array([A, B, C, D])
path = Path(polygon)

xy_points = xyz[:, :2]  # X와 Y 좌표만 사용
mask = path.contains_points(xy_points)

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

## 4. 철근 pcd xy평면 회전
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

## 5. 철근 3차원 경계상자 fitting
```python
pcd_fit, ind = pcd_rot.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

o3d.visualization.draw_geometries([pcd_fit, bb_rot])

z_values = np.asarray(pcd_fit.points)[:, 2]
z_min, z_max = z_values.min(), z_values.max()

min_bound, max_bound = bb_rot.min_bound.copy(), bb_rot.max_bound.copy()
min_bound[2], max_bound[2] = z_min, z_max

bb_fit = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
bb_fit.color = (1, 0, 0)

o3d.visualization.draw_geometries([pcd_fit, bb_fit])

# ply 파일 저장 및 확인

output_path = '/home/hyebin/lee/filtered_pcd/0904_js_007_filtered.ply'
o3d.io.write_point_cloud(output_path,pcd_fit)

pcd_filtered = o3d.io.read_point_cloud(output_path)
o3d.visualization.draw_geometries([pcd_filtered])
```
