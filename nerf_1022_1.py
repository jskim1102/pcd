


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA
import os
from PIL import Image
import cv2


mode = "nerfacto"


#%% 1. 3차원 경계상자 [pcd_crop]

path = 'N:/2024/[1]_nerfstudio/[3]_pcd/nerfacto/1011_hb_020.ply'
name = path.split('/')[5]

pcd = o3d.io.read_point_cloud(path)

bb_crop = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-5, -5, -5), max_bound=(5, 5, 5))
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

pcd_crop, bb_crop.color = pcd.crop(bb_crop), (1, 0, 0)

# o3d.visualization.draw_geometries([pcd, bb_crop])
# o3d.visualization.draw_geometries([pcd_crop, bb_crop, coord])

#%% 2. A, B, C, D 좌표 찍기 [pcd_mask]

if len(np.asarray(pcd_crop.colors)) > 0: 
    xyz_crop, rgb_crop = np.asarray(pcd_crop.points), np.asarray(pcd_crop.colors)
else:
    xyz_crop, rgb_crop = np.asarray(pcd_crop.points), []

x_crop, y_crop = xyz_crop[:, 0], xyz_crop[:, 1]

# plt.figure(figsize=(10, 8))
# plt.xticks(range(int(min(x_crop)), int(max(x_crop)) + 1, 1))
# plt.yticks(range(int(min(y_crop)), int(max(y_crop)) + 1, 1))
# plt.scatter(x_crop, y_crop, s=1, c='blue') 
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Top-down View of the Cropped Point Cloud')
# plt.grid(True)
# plt.axis('equal')
# plt.show()

A, B, C, D = (-1, 1), (-1, -1), (1, -1), (1, 1)

# plt.figure(figsize=(10, 8))
# plt.xticks(range(int(min(x_crop)), int(max(x_crop)) + 1, 1))
# plt.yticks(range(int(min(y_crop)), int(max(y_crop)) + 1, 1))
# plt.scatter(x_crop, y_crop, s=1, c='blue')  
# plt.scatter(*zip(A, B, C, D), s=100, c='red')  
# plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
# plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
# plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
# plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Top-down View of the Filtered Point Cloud (with A, B, C, D Points)')
# plt.grid(True)
# plt.axis('equal')
# plt.show()


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

# plt.figure(figsize=(10, 8))
# plt.scatter(x_mask, y_mask, s=1, c='blue')
# plt.scatter(*zip(A, B, C, D), s=100, c='red')
# plt.text(A[0]-.1, A[1]-.1, 'A', fontsize=15, color='red')
# plt.text(B[0]-.1, B[1]-.1, 'B', fontsize=15, color='red')
# plt.text(C[0]-.1, C[1]-.1, 'C', fontsize=15, color='red')
# plt.text(D[0]-.1, D[1]-.1, 'D', fontsize=15, color='red')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Top-down View of the Filtered Point Cloud (with A, B, C, D Points)')
# plt.grid(True)
# plt.axis('equal')
# plt.show()

bb_mask = pcd_mask.get_axis_aligned_bounding_box()
bb_mask.color = (1, 0, 0)  # bounding box 색상을 빨간색으로 설정

# o3d.visualization.draw_geometries([pcd_mask, bb_mask])

stand = []

# 마우스 클릭 이벤트 처리 함수
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        # 클릭한 좌표를 리스트에 추가
        stand.append((round(event.xdata, 3), round(event.ydata, 3)))

        # 클릭한 좌표에 빨간 점 찍기
        plt.scatter(event.xdata, event.ydata, c='green', s=50)
        plt.text(event.xdata, event.ydata, f'({event.xdata:.2f}, {event.ydata:.2f})', fontsize=12, color='green')
        plt.draw()


# 그래프 생성
fig, ax = plt.subplots(figsize=(10, 8))
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

# 클릭 이벤트 핸들러 등록
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# 그래프 보여주기
plt.show()

if len(stand) == 4:
    print(f"A: {stand[0]}, B: {stand[1]}, C: {stand[3]}, D: {stand[2]}")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

A, B, C, D = stand[0], stand[1], stand[2], stand[3]

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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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

#%% 4. 노이즈 제거 및 철근 구조 영역에 tight한 3차원 경계상자 추출 [pcd_fit]

if len(np.asarray(pcd_crop.colors)) > 0: 
    pcd_rot_np, pcd_rot_colors = np.asarray(pcd_rot.points), np.asarray(pcd_rot.colors)
else:
    pcd_rot_np, pcd_rot_colors = np.asarray(pcd_rot.points), []

z_values = np.unique(np.round(pcd_rot_np[:, 2], decimals=4))

filtered_points, filtered_colors = [], []

threshold = 200

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

o3d.visualization.draw_geometries([pcd_flipped, bb_flipped, coord])

filename = name.split('.')[0] + '_filtered.ply'

if not os.path.exists(os.path.join('N:/2024/[1]_nerfstudio/[4]_filtered_pcd/', mode)):
    os.makedirs(os.path.join('N:/2024/[1]_nerfstudio/[4]_filtered_pcd/', mode))

o3d.io.write_point_cloud(os.path.join('N:/2024/[1]_nerfstudio/[4]_filtered_pcd/', mode, filename), pcd_flipped)

