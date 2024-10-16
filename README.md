1002 : 3d pcd 후보 영역 지정후 crop(①)하고 a,b,c,d 좌표로 pcd 추출(②)

1003 : pcd xy평면 회전(③) 및 노이즈 제거 후 3d bbox fitting(④)

1004 : filtering된 pcd(①+②+③) 저장

1008 : 노이즈 제거 코드 수정 및 계층 분리된 height map 저장(⑤+⑥)

1010 : 회전 후 flip된 pcd 복구 코드 추가(④)

1015 : 전체적으로 코드 수정(nerfstudio, gaussian_splatting 모두 동작 및 pcd_crop -> pcd_masak -> pcd_rot -> pcd_fit -> pcd_flipped -> height map 코드 모두 정리)

# New Code [nerf_1015]
## 🄋 필요한 함수 정의
```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA
import os
from PIL import Image


def color_scale(PCD):
    z_vals = np.asarray(PCD.points)[:, 2]
    norm = plt.Normalize(vmin=min(z_vals), vmax=max(z_vals))
    colormap = plt.get_cmap('viridis') 
    colors = colormap(norm(z_vals))[:, :3] 
    PCD.colors = o3d.utility.Vector3dVector(colors)
    return PCD

def height_map_img(points, path):   
    
    grid_size = 0.001
    
    # 포인트 클라우드에서 X, Y, Z 추출
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # x, y 축의 그리드를 만듭니다.
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    
    # 높이 맵을 흰색 배경으로 초기화 (1.0 -> 흰색: 255)
    height_map = np.full((len(y_grid), len(x_grid)), 255)  # 흰색 배경 (255)
    
    # 포인트들을 XY grid에 맵핑하여 Z 값을 height map으로 변환
    for point in points:
        x_idx = int((point[0] - x_min) / grid_size)
        y_idx = int((point[1] - y_min) / grid_size)
        if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
            height_map[y_idx, x_idx] = 0  # Z 값이 있는 경우 검은색(0)으로 변환
    
    # 이미지를 생성 (uint8 형식으로 변환)
    img = Image.fromarray(np.uint8(height_map), 'L')
    
    return img

def height_map_np(points, path): # 넘파이 height map
    
    grid_size = 0.001
    
    # 포인트 클라우드에서 X, Y, Z 추출
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    x_grid, y_grid = np.arange(x_min, x_max, grid_size), np.arange(y_min, y_max, grid_size)
    
    height_map = np.full((len(y_grid), len(x_grid)), np.nan)
    
    for point in points:
        x_idx = int((point[0] - x_min) / grid_size)
        y_idx = int((point[1] - y_min) / grid_size)
        if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
            height_map[y_idx, x_idx] = point[2]

    min_height = np.nanmin(height_map)
    height_map = np.nan_to_num(height_map, nan=min_height)
    
    return height_map
```
## ① ply 파일 읽어서 3d pcd의 후보 영역(3d bbox)을 지정하고 좌표축과 함께 시각화(가우시안은 좌표축이 다르기 때문에 nerf와 같아지도록 처리)
```python
#%% 1. 3차원 경계상자 [pcd_crop]

path = 'N:/2024/[1]_kjs_lee/[3]_pcd/1011_js_018.ply'
# path = 'N:/2024/[2]_gaussian-splatting/data/1011_js_004/output/da612658-d/point_cloud/iteration_30000/point_cloud.ply'
pcd = o3d.io.read_point_cloud(path)

bb_crop = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-5, -5, -5), max_bound=(5, 5, 5))
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])


if 'splatting' in path:
    rotation_angle = np.pi / 2
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [0, np.sin(rotation_angle), np.cos(rotation_angle)]])

    xyz_ = np.asarray(pcd.points) @ rotation_matrix
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(xyz_)
    pcd = pcd_
    

pcd_crop, bb_crop.color = pcd.crop(bb_crop), (1, 0, 0)

# o3d.visualization.draw_geometries([pcd, bb_crop])
o3d.visualization.draw_geometries([pcd_crop, bb_crop, coord])
```

## ② 3d pcd를 수직 위에서 바라본 2차원 BEV 이미지로 시각화하기 -> x, y 좌표값을 확인할수 있음(확인한 좌표값에 따라 철근 구조물에 해당하는 pcd를 추출하기 위한 폴리곤 좌표 A, B, C, D를 설정) -> A, B, C, D 좌표값의 폴리곤 내부의 pcd만 추출
```python
#%% 2. A, B, C, D 좌표 찍기 [pcd_mask]

if len(np.asarray(pcd_crop.colors)) > 0: 
    xyz_crop, rgb_crop = np.asarray(pcd_crop.points), np.asarray(pcd_crop.colors)
else:
    xyz_crop, rgb_crop = np.asarray(pcd_crop.points), []

x_crop, y_crop = xyz_crop[:, 0], xyz_crop[:, 1]

plt.figure(figsize=(10, 8))
plt.xticks(range(int(min(x_crop)), int(max(x_crop)) + 1, 1))
plt.yticks(range(int(min(y_crop)), int(max(y_crop)) + 1, 1))
plt.scatter(x_crop, y_crop, s=1, c='blue') 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Top-down View of the Cropped Point Cloud')
plt.grid(True)
plt.axis('equal')
plt.show()

# A, B, C, D = (-1, 1), (-1, -1), (1, -1), (1, 1)
# A, B, C, D = (-4, 4), (-4, -4), (4, -4), (4, 4)
A, B, C, D = (-.77, -0.15), (-0.06, -0.79), (.82, 0.15), (0.10, 0.84)

plt.figure(figsize=(10, 8))
plt.xticks(range(int(min(x_crop)), int(max(x_crop)) + 1, 1))
plt.yticks(range(int(min(y_crop)), int(max(y_crop)) + 1, 1))
plt.scatter(x_crop, y_crop, s=1, c='blue')  
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


polygon = np.array([A, B, C, D])
matplot_path_ = Path(polygon)

xy_crop = xyz_crop[:, :2]
mask = matplot_path_.contains_points(xy_crop)

if len(np.asarray(pcd_crop.colors)) > 0: 
    xyz_mask, rgb_mask = xyz_crop[mask], rgb_crop[mask]
    
    pcd_mask = o3d.geometry.PointCloud()
    pcd_mask.points, pcd_mask.colors = o3d.utility.Vector3dVector(xyz_mask), o3d.utility.Vector3dVector(rgb_mask)    
else:
    xyz_mask, rgb_mask = xyz_crop[mask], []
    
    pcd_mask = o3d.geometry.PointCloud()
    pcd_mask.points = o3d.utility.Vector3dVector(xyz_mask)
    
x_mask, y_mask = xyz_mask[:, 0], xyz_mask[:, 1]

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

bb_mask = pcd_mask.get_axis_aligned_bounding_box()
bb_mask.color = (1, 0, 0)  # bounding box 색상을 빨간색으로 설정

o3d.visualization.draw_geometries([pcd_mask, bb_mask])
```

## ③ 철근 pcd xy평면 회전
```python
#%% 3. 주성분 분석(PCA)을 활용한 xy평면 회전 [pcd_rot]

pca = PCA(n_components=3)
pca.fit(pcd_mask.points)

matrix_rot = pca.components_.T
xyz_rot = pcd_mask.points @ matrix_rot

if len(np.asarray(pcd_crop.colors)) > 0: 
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points, pcd_rot.colors = o3d.utility.Vector3dVector(xyz_rot), pcd_mask.colors
else:
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(xyz_rot)

bb_rot = pcd_rot.get_axis_aligned_bounding_box()
bb_rot.color = (1, 0, 0)

o3d.visualization.draw_geometries([pcd_rot, bb_rot])
```

## ④ xy평면의 point 개수가 임계값 이하인 z층 point를 삭제하여 noise 제거 -> 3차원 경계상자 fitting
```python
#%% 4. 노이즈 제거 및 철근 구조 영역에 tight한 3차원 경계상자 추출 [pcd_fit]

if len(np.asarray(pcd_crop.colors)) > 0: 
    pcd_rot_np, pcd_rot_colors = np.asarray(pcd_rot.points), np.asarray(pcd_rot.colors)
else:
    pcd_rot_np, pcd_rot_colors = np.asarray(pcd_rot.points), []

z_values = np.unique(np.round(pcd_rot_np[:, 2], decimals=4))

filtered_points, filtered_colors = [], []

threshold = 20

for z in z_values:
    # 특정 xy평면 선택
    filtered = np.abs(pcd_rot_np[:, 2] - z) < 1e-4
    
    if len(np.asarray(pcd_crop.colors)) > 0: 
        points_in_slice, colors_in_slice = pcd_rot_np[filtered], pcd_rot_colors[filtered]
        
        # 특정 xy평면에 임계값 이상의 pcd가 있는 경우 (개수)
        if len(points_in_slice) >= threshold:
            filtered_points.append(points_in_slice)
            filtered_colors.append(colors_in_slice)
    else:
        points_in_slice, colors_in_slice = pcd_rot_np[filtered], []
        
        # 특정 xy평면에 임계값 이상의 pcd가 있는 경우 (개수)
        if len(points_in_slice) >= threshold:
            filtered_points.append(points_in_slice)

if len(np.asarray(pcd_crop.colors)) > 0: 
    filtered_point_cloud, filtered_colors_cloud = np.vstack(filtered_points), np.vstack(filtered_colors)
    
    # Open3D 포인트 클라우드 객체 생성 (필터링된 데이터)
    pcd_fit = o3d.geometry.PointCloud()
    pcd_fit.points, pcd_fit.colors = o3d.utility.Vector3dVector(filtered_point_cloud), o3d.utility.Vector3dVector(filtered_colors_cloud)

else:
    filtered_point_cloud, filtered_colors_cloud = np.vstack(filtered_points), []
    
    # Open3D 포인트 클라우드 객체 생성 (필터링된 데이터)
    pcd_fit = o3d.geometry.PointCloud()
    pcd_fit.points = o3d.utility.Vector3dVector(filtered_point_cloud)

bb_fit = pcd_fit.get_axis_aligned_bounding_box()
bb_fit.color = (1, 0, 0)

o3d.visualization.draw_geometries([pcd_fit, bb_fit])
```

## ⑤ 회전 후 flip된 pcd 복구
```python
#%% 5. z flip [pcd_flipped]

pcd_fit_np = np.copy(np.asarray(pcd_fit.points))
pcd_fit_np[:, 2] *= -1

if len(np.asarray(pcd_crop.colors)) > 0:
    pcd_flipped = o3d.geometry.PointCloud()
    pcd_flipped.points, pcd_flipped.colors = o3d.utility.Vector3dVector(pcd_fit_np), pcd_fit.colors
else:
    pcd_flipped = o3d.geometry.PointCloud()
    pcd_flipped.points = o3d.utility.Vector3dVector(pcd_fit_np)


min_bound_np, max_bound_np = np.copy(np.asarray(bb_fit.min_bound)), np.copy(np.asarray(bb_fit.max_bound))

min_bound_np[2] *= -1
max_bound_np[2] *= -1

bb_flipped = o3d.geometry.AxisAlignedBoundingBox(min_bound_np, max_bound_np)
bb_flipped.color = (1, 0, 0)

o3d.visualization.draw_geometries([pcd_flipped, bb_flipped])

name = path.split('/')[4]
if 'splatting' not in path:
    filename = name.split('.')[0] + '_filtered.ply'
    o3d.io.write_point_cloud(os.path.join('N:/2024/[1]_kjs_lee/[4]_filtered_pcd/', filename), pcd_flipped)
else:
    filename = name.split('.')[0]  + '_filtered_gaussian.ply'
    o3d.io.write_point_cloud(os.path.join('N:/2024/[2]_gaussian-splatting/[4]_filtered_pcd/', filename), pcd_flipped)
```

## ⑥ Z값을 기준으로 계층 분리 후 Height Map 출력 및 저장
```python
#%% 6. Height Map

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA
import os
from PIL import Image

pcd_final = pcd_flipped


output_path = 'N:/2024/[1]_kjs_lee/[4]_filtered_pcd/1011_js_016_filtered.ply'
# output_path = 'N:/2024/[2]_gaussian-splatting/[4]_filtered_pcd/1011_js_002_filtered_gaussian.ply'
# pcd_final = o3d.io.read_point_cloud(output_path)


coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_final, coord])

final_xyz = np.asarray(pcd_final.points)

z_vals = final_xyz[:, 2]  # Z 값 추출
z_min, z_max = z_vals.min(), z_vals.max()

z_thresh = z_max - (abs(z_min) - abs(z_max)) * 0.177 # 1층, 2층 분리

layer_1_mask = z_vals <= z_thresh
layer_2_mask = z_vals > z_thresh

layer_1_xyz, layer_2_xyz = final_xyz[layer_1_mask], final_xyz[layer_2_mask]

layer_1_pcd, layer_2_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
layer_1_pcd.points, layer_2_pcd.points = o3d.utility.Vector3dVector(layer_1_xyz), o3d.utility.Vector3dVector(layer_2_xyz)

# o3d.visualization.draw_geometries([layer_1_pcd])
# o3d.visualization.draw_geometries([layer_2_pcd])

if 'splatting' not in path:    
    def capture_image_1(vis):
        vis.capture_screen_image('N:/2024/[1]_kjs_lee/[5]_height_map/' + name.split('.')[0] + "_layer_1_o3d.png") 
        return False
    
    def capture_image_2(vis):
        vis.capture_screen_image('N:/2024/[1]_kjs_lee/[5]_height_map/' + name.split('.')[0] + "_layer_2_o3d.png") 
        return False
else:
    
    def capture_image_1(vis):
        vis.capture_screen_image('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name.split('.')[0] + "_layer_1_o3d_gaussian.png") 
        return False  
    
    def capture_image_2(vis):
        vis.capture_screen_image('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name.split('.')[0] + "_layer_2_o3d_gaussian.png") 
        return False  
    
key_to_callback = {}
key_to_callback[ord("S")]= capture_image_1
o3d.visualization.draw_geometries_with_key_callbacks([layer_1_pcd], key_to_callback)

key_to_callback = {}
key_to_callback[ord("S")]= capture_image_2
o3d.visualization.draw_geometries_with_key_callbacks([layer_2_pcd], key_to_callback)

# height_map 만들기
layer_1_img , layer_2_img = height_map_img(layer_1_xyz, path), height_map_img(layer_2_xyz, path)
layer_1_np , layer_2_np = height_map_np(layer_1_xyz, path), height_map_np(layer_2_xyz, path)

if 'splatting' not in path:
    
    layer_1_img.save('N:/2024/[1]_kjs_lee/[5]_height_map/' + name + "_layer_1_img.png")
    layer_2_img.save('N:/2024/[1]_kjs_lee/[5]_height_map/' + name + "_layer_2_img.png")

    plt.imshow(layer_1_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[1]_kjs_lee/[5]_height_map/' + name + "_layer_1_np.png", layer_1_np, cmap='viridis')
    
    plt.imshow(layer_2_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[1]_kjs_lee/[5]_height_map/' + name + "_layer_2_np.png", layer_1_np, cmap='viridis')

else:
    
    layer_1_img.save('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_1_img_gaussian.png")
    layer_2_img.save('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_2_img_gaussian.png")

    plt.imshow(layer_1_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_1_np_gaussian.png", layer_1_np, cmap='viridis')
    
    plt.imshow(layer_2_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_2_np_gaussian.png", layer_1_np, cmap='viridis')
```
