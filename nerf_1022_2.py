
import os
import f_dpi as dpi
import copy
import open3d as o3d
import numpy as np 
import matplotlib.pyplot as plt
import f_dpi as dpi
import cv2
from PIL import Image

def click_event(event, x, y, flags, param):
    global stand_new

    # 왼쪽 마우스 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 좌표를 리스트에 추가
        stand_new.append((x, y))

        # 좌표에 빨간 점 찍기
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

        # 네 개의 좌표가 모두 입력되면 폴리곤 그리기
        if len(stand_new) == 4:
            # 폴리곤 그리기
            pts = np.array(stand_new, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow('Image', img)


mode = "nerfacto"

path0 = os.path.join('N:/2024/[1]_nerfstudio/[4]_filtered_pcd/', mode)

if not os.path.exists(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode)):
    os.makedirs(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode))

if not os.path.exists(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_rot_b/', mode)):
    os.makedirs(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_rot_b/', mode))
    
paths0 = os.listdir(path0)

paths0 = paths0[92:117]

#%% 1

for path0_ in paths0:

    pcd_final = o3d.io.read_point_cloud(os.path.join(path0, path0_))

    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_final, coord])
    
    final_xyz = np.asarray(pcd_final.points)
    
    z_vals = final_xyz[:, 2]  # Z 값 추출
    z_min, z_max = z_vals.min(), z_vals.max()
    
    z_thresh = z_max - (abs(z_min) - abs(z_max)) * 0.177 # 1층, 2층 분리
    
    layer_1_mask = z_vals <= z_thresh
    layer_2_mask = z_vals > z_thresh
    
    layer_1_xyz, layer_2_xyz = final_xyz[layer_1_mask], final_xyz[layer_2_mask]
    
    layer_1_pcd, layer_2_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    layer_1_pcd.points, layer_2_pcd.points = o3d.utility.Vector3dVector(layer_1_xyz), o3d.utility.Vector3dVector(layer_2_xyz)
    
    
    xy_b_layer_1, xy_b_layer_2 = layer_1_xyz[:, :2], layer_2_xyz[:, :2]
    
    
    x_b_layer1, y_b_layer1 = xy_b_layer_1[:, 0], xy_b_layer_1[:, 1]
    x_b_layer2, y_b_layer2 = xy_b_layer_2[:, 0], xy_b_layer_2[:, 1]
    
    
    plt.figure(figsize=(18, 12))
    plt.scatter(x_b_layer1, y_b_layer1, s=1, c='black')
    plt.savefig(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_b.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.figure(figsize=(18, 12))
    plt.scatter(x_b_layer2, y_b_layer2, s=1, c='black')
    plt.savefig(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    img_layer_1 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_b.png")
    img_layer_1 = cv2.resize(img_layer_1, (0, 0), fx=0.4, fy=0.4)
    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_b.png", img_layer_1)
    
    img_layer_2 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png")
    img_layer_2 = cv2.resize(img_layer_2, (0, 0), fx=0.4, fy=0.4)
    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png", img_layer_2)
    
    stand_new = []
    
    img = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png")
    
    oimg = copy.deepcopy(img)    
    # 이미지 창 열기
    cv2.imshow('Image', img)
    
    # 마우스 이벤트 등록
    cv2.setMouseCallback('Image', click_event)
    
    # ESC 키를 누르면 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4개의 좌표 (예시: 좌상단, 우상단, 우하단, 좌하단 순서로)
    pts = np.array([stand_new], dtype="float32")
    
    # 크롭할 영역의 크기를 결정 (출력할 이미지의 크기)
    width, height = oimg.shape[1], oimg.shape[0]
    
    # 새로운 사각형 좌표 (크롭 후의 이미지가 직사각형이 되도록 설정)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    # Perspective 변환 행렬 계산
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    
    img_layer1 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_b.png")
    img_layer2 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png")
    
    warped_layer1 = cv2.warpPerspective(img_layer1, M, (width, height))
    warped_layer2 = cv2.warpPerspective(img_layer2, M, (width, height))
    

    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[6]_height_map_rot_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_b.png", warped_layer1)
    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[6]_height_map_rot_b/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_b.png", warped_layer2)

#%% 2

for path0_ in paths0:

    pcd_final = o3d.io.read_point_cloud(os.path.join(path0, path0_))

    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_final, coord])
    
    final_xyz = np.asarray(pcd_final.points)
    
    z_vals = final_xyz[:, 2]  # Z 값 추출
    z_min, z_max = z_vals.min(), z_vals.max()
    
    z_thresh = z_max - (abs(z_min) - abs(z_max)) * 0.177 # 1층, 2층 분리
    
    layer_1_mask = z_vals <= z_thresh
    layer_2_mask = z_vals > z_thresh
    
    layer_1_xyz, layer_2_xyz = final_xyz[layer_1_mask], final_xyz[layer_2_mask]
    
    layer_1_pcd, layer_2_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    layer_1_pcd.points, layer_2_pcd.points = o3d.utility.Vector3dVector(layer_1_xyz), o3d.utility.Vector3dVector(layer_2_xyz)
    
    
    grid_a = 0.001

    x_min_a, x_max_a = layer_1_xyz[:, 0].min(), layer_1_xyz[:, 0].max()
    y_min_a, y_max_a = layer_1_xyz[:, 1].min(), layer_1_xyz[:, 1].max()

    x_grid_a, y_grid_a = np.arange(x_min_a, x_max_a, grid_a), np.arange(y_min_a, y_max_a, grid_a)

    height_map_a_layer1 = np.full((len(y_grid_a), len(x_grid_a)), 255)

    for point in layer_1_xyz:
        x_idx_a = int((point[0] - x_min_a) / grid_a)
        y_idx_a = int((point[1] - y_min_a) / grid_a)
        if 0 <= x_idx_a < len(x_grid_a) and 0 <= y_idx_a < len(y_grid_a):
            height_map_a_layer1[y_idx_a, x_idx_a] = 0

    img_a_layer1 = Image.fromarray(np.uint8(height_map_a_layer1), 'L')
    
    x_min_a, x_max_a = layer_2_xyz[:, 0].min(), layer_2_xyz[:, 0].max()
    y_min_a, y_max_a = layer_2_xyz[:, 1].min(), layer_2_xyz[:, 1].max()

    x_grid_a, y_grid_a = np.arange(x_min_a, x_max_a, grid_a), np.arange(y_min_a, y_max_a, grid_a)

    height_map_a_layer2 = np.full((len(y_grid_a), len(x_grid_a)), 255)

    for point in layer_2_xyz:
        x_idx_a = int((point[0] - x_min_a) / grid_a)
        y_idx_a = int((point[1] - y_min_a) / grid_a)
        if 0 <= x_idx_a < len(x_grid_a) and 0 <= y_idx_a < len(y_grid_a):
            height_map_a_layer2[y_idx_a, x_idx_a] = 0

    img_a_layer2 = Image.fromarray(np.uint8(height_map_a_layer2), 'L')
    
    img_a_layer1.save(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_a.png")
    img_a_layer2.save(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_a.png")
    
    
    
    stand_new = []
    
    img = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_a.png")
    
    oimg = copy.deepcopy(img)    
    # 이미지 창 열기
    cv2.imshow('Image', img)
    
    # 마우스 이벤트 등록
    cv2.setMouseCallback('Image', click_event)
    
    # ESC 키를 누르면 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4개의 좌표 (예시: 좌상단, 우상단, 우하단, 좌하단 순서로)
    pts = np.array([stand_new], dtype="float32")
    
    # 크롭할 영역의 크기를 결정 (출력할 이미지의 크기)
    
    # 새로운 사각형 좌표 (크롭 후의 이미지가 직사각형이 되도록 설정)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    # Perspective 변환 행렬 계산
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    
    img_layer1 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_a.png")
    img_layer2 = dpi.imgload(os.path.join('N:/2024/[1]_nerfstudio/[5]_height_map_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_a.png")
    
    warped_layer1 = cv2.warpPerspective(img_layer1, M, (width, height))
    warped_layer2 = cv2.warpPerspective(img_layer2, M, (width, height))
    

    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[6]_height_map_rot_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_1_img_a.png", warped_layer1)
    dpi.imwrite(os.path.join('N:/2024/[1]_nerfstudio/[6]_height_map_rot_a/', mode) + '/' + path0_.split('_filtered')[0] + "_layer_2_img_a.png", warped_layer2)



#%% Biliateral filter

# # 그리드 해상도 설정 (필요에 따라 조정 가능)
# grid_c = 0.005  # 1cm 간격으로 그리드를 만듦
# # Point cloud 데이터 (예제 데이터로 xyz_final 사용)
# Pts_c = layer_2_xyz  # Pts는 (x, y, z) 좌표로 이루어진 포인트 클라우드

# # X와 Y 값만 추출
# x_c, y_c, z_c = Pts_c[:, 0], Pts_c[:, 1], Pts_c[:, 2]

# # X와 Y의 최소, 최대값을 구해서 그리드의 범위를 결정
# x_min_c, x_max_c = np.min(x_c), np.max(x_c)
# y_min_c, y_max_c = np.min(y_c), np.max(y_c)

# # 그리드 사이즈 계산
# x_bins_c = np.arange(x_min_c, x_max_c, grid_c)
# y_bins_c = np.arange(y_min_c, y_max_c, grid_c)

# # X, Y 그리드에 맞춰 Z 값을 평균으로 계산
# grid_x_c, grid_y_c = np.meshgrid(x_bins_c, y_bins_c)

# # 그리드에 들어갈 height 값 (Z 값)을 초기화
# height_map_c = np.full(grid_x_c.shape, np.nan)

# # 각 그리드 셀에 Z 값 할당
# for i in range(len(x_bins_c) - 1):
#     for j in range(len(y_bins_c) - 1):
#         # 현재 그리드 셀 안에 포함되는 포인트 찾기
#         mask = (x_c >= x_bins_c[i]) & (x_c < x_bins_c[i + 1]) & \
#                (y_c >= y_bins_c[j]) & (y_c < y_bins_c[j + 1])
        
#         # 해당 그리드에 포함된 포인트들의 Z 값의 평균을 height로 설정
#         if np.any(mask):
#             height_map_c[j, i] = np.mean(z_c[mask])

# img_c_layer2 = np.nan_to_num(height_map_c)
# img_c_layer2 = np.where(img_c_layer1 == 0, 255, 0)

# img_c_layer2 = Image.fromarray(np.uint8(img_c_layer2), 'L')

# # plt.figure(figsize=(10, 10))
# # plt.imshow(height_map_b, extent=[x_min_b, x_max_b, y_min_b, y_max_b], origin='lower', cmap='hsv')
# # plt.colorbar(label='Height (Z-axis)')
# # plt.title('Height Map (Bird\'s Eye View)')
# # plt.xlabel('X Axis')
# # plt.ylabel('Y Axis')
# # plt.show()


        