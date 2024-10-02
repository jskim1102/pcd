# nerf_1002
## 1. ply 파일 읽어서 3d pcd의 영역을 지정하기(bb내부 영역에 있는 pcd만 추출 : pcd_crop)
```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

path = 'N:/2024/[1]_kjs_lee/0930_js_004.ply'
pcd = o3d.io.read_point_cloud(path)

# o3d.visualization.draw_geometries([pcd])

bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(1, 1, .1))
pcd_crop = pcd.crop(bb)
bb.color = (1, 0, 0) 

# o3d.visualization.draw_geometries([pcd, bb])
o3d.visualization.draw_geometries([pcd_crop, bb])
```


## 2. 3d pcd를 수직 위에서 바라본 2차원 BEV 이미지로 시각화하기 -> x, y 좌표값을 확인할수 있음(확인한 좌표값에 따라 철근 구조물에 해당하는 pcd를 추출하기 위한 폴리곤 좌표 A, B, C, D를 설정)
```python
xyz, rgb = np.asarray(pcd_crop.points), np.asarray(pcd_crop.colors)

x = xyz[:, 0]
y = xyz[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, c='blue')  # s is the marker size
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Cropped Point Cloud')
plt.grid(True)
plt.axis('equal')
plt.show()


A = (-0.75, 0.8)
B = (-0.5, -0.5)
C = (0.5, -0.3)
D = (0.2, 1.0)


plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, c='blue')  # 포인트 클라우드
plt.scatter(*zip(A, B, C, D), s=100, c='red')  # A, B, C, D 점을 빨간색으로 표시
plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Filtered Point Cloud (with A, B, C, D Points)')
plt.grid(True)
plt.axis('equal')  # 비율 유지
plt.show()
```

## 3. A, B, C, D 좌표값의 폴리곤 내부의 pcd만 추출
```python
polygon = np.array([A, B, C, D])

path = Path(polygon)

xy_points = xyz[:, :2]
mask = path.contains_points(xy_points)

xyz_in, rgb_in = xyz[mask], rgb[mask]
pcd_in = o3d.geometry.PointCloud()
pcd_in.points = o3d.utility.Vector3dVector(xyz_in)
pcd_in.colors = o3d.utility.Vector3dVector(rgb_in)

o3d.visualization.draw_geometries([pcd_in])

x_in = xyz_in[:, 0]
y_in = xyz_in[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(x_in, y_in, s=1, c='blue') 
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
