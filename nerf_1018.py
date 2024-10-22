

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA
import os
from PIL import Image
import cv2
import f_dpi as dpi
from copy import deepcopy
import copy

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

def click_event(event, x, y, flags, param):
    global stand_new

    # 왼쪽 마우스 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 좌표를 리스트에 추가
        stand_new.append((x, y))

        # 좌표에 빨간 점 찍기
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', image)

        # 네 개의 좌표가 모두 입력되면 폴리곤 그리기
        if len(stand_new) == 4:
            # 폴리곤 그리기
            pts = np.array(stand_new, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow('Image', image)

#%% 1. 3차원 경계상자 [pcd_crop]

path = 'N:/2024/[1]_nerfstudio/[3]_pcd/splatfacto/1011_js_004.ply'
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

A, B, C, D = (-1, 1), (-1, -1), (1, -1), (1, 1)
# A, B, C, D = (-0.541, 0.611), (-0.62, -0.321),  (0.587, -0.446), (0.677, 0.479)
 

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

name = path.split('/')[5]
if 'splatting' not in path:
    filename = name.split('.')[0] + '_filtered.ply'
    o3d.io.write_point_cloud(os.path.join('N:/2024/[1]_nerfstudio/[4]_filtered_pcd/splatfacto/', filename), pcd_flipped)
else:
    filename = name.split('.')[0]  + '_filtered_gaussian.ply'
    o3d.io.write_point_cloud(os.path.join('N:/2024/[2]_nerfstudio/[4]_filtered_pcd/splatfacto/', filename), pcd_flipped)


#%% 6. Height Map

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from sklearn.decomposition import PCA
import os
from PIL import Image

pcd_final = pcd_flipped

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
        vis.capture_screen_image('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name.split('.')[0] + "_layer_1_o3d.png") 
        return False
    
    def capture_image_2(vis):
        vis.capture_screen_image('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name.split('.')[0] + "_layer_2_o3d.png") 
        return False
# else:
    
#     def capture_image_1(vis):
#         vis.capture_screen_image('N:/2024/[2]_gaussian-splatting/[5]_height_map/splatfacto/' + name.split('.')[0] + "_layer_1_o3d_gaussian.png") 
#         return False  
    
#     def capture_image_2(vis):
#         vis.capture_screen_image('N:/2024/[2]_gaussian-splatting/[5]_height_map/splatfacto/' + name.split('.')[0] + "_layer_2_o3d_gaussian.png") 
#         return False  
    
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
    
    layer_1_img.save('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_1_img.png")
    layer_2_img.save('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_2_img.png")

    plt.imshow(layer_1_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_1_np.png", layer_1_np, cmap='viridis')
    
    plt.imshow(layer_2_np, cmap='viridis')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave('N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_2_np.png", layer_1_np, cmap='viridis')

# else:
    
#     layer_1_img.save('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_1_img_gaussian.png")
#     layer_2_img.save('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_2_img_gaussian.png")

#     plt.imshow(layer_1_np, cmap='viridis')
#     plt.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.imsave('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_1_np_gaussian.png", layer_1_np, cmap='viridis')
    
#     plt.imshow(layer_2_np, cmap='viridis')
#     plt.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.imsave('N:/2024/[2]_gaussian-splatting/[5]_height_map/' + name + "_layer_2_np_gaussian.png", layer_1_np, cmap='viridis')

stand_new = []

path = 'N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_1_img.png"
path = 'N:/2024/[1]_nerfstudio/[5]_height_map/splatfacto/' + name + "_layer_2_img.png"


image = dpi.imgload(path)
oimg = copy.deepcopy(image)

filename = path.split('/')[-1]

# 이미지 창 열기
cv2.imshow('Image', image)

# 마우스 이벤트 등록
cv2.setMouseCallback('Image', click_event)

# ESC 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4개의 좌표 (예시: 좌상단, 우상단, 우하단, 좌하단 순서로)
pts = np.array([stand_new], dtype="float32")

# 크롭할 영역의 크기를 결정 (출력할 이미지의 크기)
width = oimg.shape[1]
height = oimg.shape[0]

# 새로운 사각형 좌표 (크롭 후의 이미지가 직사각형이 되도록 설정)
dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]], dtype="float32")

# Perspective 변환 행렬 계산
M = cv2.getPerspectiveTransform(pts, dst_pts)

# 이미지를 변환 및 크롭
warped = cv2.warpPerspective(oimg, M, (width, height))

savepath1 = 'N:/2024/[1]_nerfstudio/[6]_height_map_rot/splatfacto/'
savepath2 = 'N:/2024/[1]_nerfstudio/[6]_height_map_rot/splatfacto/'

dpi.imwrite(savepath1 + filename, warped)
dpi.imwrite(savepath2 + filename, warped)