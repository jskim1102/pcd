1002 : 3d pcd 후보 영역 지정후 crop(①)하고 a,b,c,d 좌표로 pcd 추출(②)

1003 : pcd xy평면 회전(③) 및 노이즈 제거 후 3d bbox fitting(④)

1004 : filtering된 pcd(①+②+③) 저장

1008 : 노이즈 제거 코드 수정 및 계층 분리된 height map 저장(⑤+⑥)

1010 : 회전 후 flip된 pcd 복구 코드 추가(④)

1015 : 전체적으로 코드 수정(nerfstudio, gaussian_splatting 모두 동작 및 pcd_crop -> pcd_mask -> pcd_rot -> pcd_fit -> pcd_flipped -> height map 코드 모두 정리)

1018 : 전체적으로 코드 수정 2차

1022(nerf_1) : [1]_nerfstudio/[4]_filtered_pcd => pcd_crop -> pcd_mask -> pcd_rot -> pcd_fit -> pcd_flipped (파일 1개씩 실행)
1022(nerf_2) : [1]_nerfstudio/[5]_height_map_a, [5]_height_map_b => height_map (for문으로 전체 파일 한번에 실행)
