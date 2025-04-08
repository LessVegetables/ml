import trimesh
import numpy as np
import matplotlib.pyplot as plt
import json


mesh_path = r'pcl_test\test3_mesh.glb'
mesh = trimesh.load(mesh_path, force='mesh')

true_height = 192
scale_factor = true_height / mesh.extents[1]
mesh.apply_scale(scale_factor)
print('\tWidth\tHeight\tThickness\n', mesh.extents)
# mesh.show()
pcl = trimesh.points.PointCloud(mesh.vertices)
pcl.colors = [[255, 255, 255, 255] for i in range(len(pcl.vertices))]


json_kp_path = r'pcl_test\input2_egor_keypoints.json'
kp = [[], []]
with open(json_kp_path, 'r') as file:
    data = json.load(file)
    input_kp = data['people'][0]['pose_keypoints_2d']
#     print("input_kp = ", input_kp)
    for i in range(1, len(input_kp) + 1):
        if(i % 3 != 0):
            if(i % 3 == 1):
                kp[0].append(input_kp[i - 1])
            else:
                kp[1].append(input_kp[i - 1])
            
print("kp = ", kp)
print(f'Min X: {min(kp[0])}. Max X: {max(kp[0])}.\nMin Y: {min(kp[1])}. Max Y: {max(kp[1])}')
print(f'Min X in mesh: {np.min(mesh.vertices[:, 0])}. Max X in mesh: {np.max(mesh.vertices[:, 0])}.\nMin Y in mesh: {np.min(mesh.vertices[:, 1])}. Max Y in mesh: {np.max(mesh.vertices[:, 1])}')

def scale_skeleton(mesh, kp):
    kp = np.array(kp).T  # Преобразуем из формата [[x1,x2,...], [y1,y2,...]] в массив формы (N,2)
    
    # Если координаты из изображения, инвертируем ось Y
    kp[:, 1] = -kp[:, 1]
    
    # Центрируем скелет: вычитаем среднее значение
    kp_centered = kp - np.mean(kp, axis=0)
    
    # Размер скелета (по X и Y)
    skel_size = np.max(kp_centered, axis=0) - np.min(kp_centered, axis=0)
    # Размер модели (по X и Y)
    mesh_size = mesh.extents[:2]
    
    # Коэффициенты масштабирования для каждой оси
    scale = mesh_size / (skel_size * 1.05)
    # Применяем масштабирование
    kp_scaled = kp_centered * scale
    
    # Смещаем скелет так, чтобы его центр совпадал с центром модели
    kp_shifted = kp_scaled + mesh.centroid[:2]
    
    # Возвращаем результат в виде списка: [[x1,x2,...], [y1,y2,...]]
    return kp_shifted.T.tolist()

# Применяем функцию
kp = scale_skeleton(mesh, kp)

def draw_skeleton_2d(kp, connections=None):
    x, y = kp
    # plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='r')

    if connections:
        for i, j in connections:
            plt.plot([x[i], x[j]], [y[i], y[j]], 'b-')

    plt.gca().invert_yaxis()  # Совместимость с OpenPose
    plt.axis('equal')
    plt.title("2D скелет")
    plt.show()

# connections — список ребер скелета, например:
connections = [
#     (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (15, 17), (0, 15), (0, 16),
#     (16, 18), (1, 5), (1, 8), (8, 12), (21, 19), (21, 20), (21, 14), (24, 22), (24, 23), (24, 11), (0, 1)
    (2, 3)
]

# draw_skeleton_2d(kp, connections)

kp = np.array(kp)
z_cord = np.array([-7 for i in range(kp.shape[1])]).reshape(1, -1)
kp = np.concatenate((kp, z_cord), axis=0)
print(kp.shape)
kp3d = kp.T

kp3d = np.array((kp3d[:, 0], kp3d[:, 1], kp3d[:, 2])).T
# print(f'z cord = {z_cord}')
# print(f'Kp3d z = {kp3d[:, 2]}')

pcl_kp3d = trimesh.points.PointCloud(kp3d, colors=[[0, 255, 0, 255] for i in range(len(kp3d))])
# print(f'Pcl shape = {type(pcl.vertices)}, Kp3d shape = {kp3d.shape}')

print(kp3d.shape, pcl_kp3d.shape, pcl_kp3d.extents, pcl.extents)
scene = trimesh.Scene(geometry=pcl_kp3d)
scene.add_geometry(geometry=pcl)
scene.show()