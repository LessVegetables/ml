import bpy
import json
import mathutils

# === 1. Загружаем OpenPose keypoints ===
with open("/Users/xd/Desktop/nsu_2course/2_sem/AI_PROJ/input2_egor_keypoints.json") as f:
    data = json.load(f)

raw_keypoints = data["people"][0]["pose_keypoints_2d"]
points = [(raw_keypoints[i], raw_keypoints[i+1], raw_keypoints[i+2]) for i in range(0, len(raw_keypoints), 3)]
valid_points = {i: (x / 1000.0, 0, -y / 1000.0) for i, (x, y, conf) in enumerate(points) if conf > 0.5}

# === 2. Создаем empties ===
empties = {}
for i, loc in valid_points.items():
    empty = bpy.data.objects.new(f"joint_{i}", None)
    bpy.context.collection.objects.link(empty)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.location = loc
    empties[i] = empty

# === 3. Соединяем в скелет (COCO формат) ===
skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (11, 22), (22, 23),
    (14, 19), (19, 20),
    (0, 15), (15, 17),
    (0, 16), (16, 18)
]

for start, end in skeleton:
    if start in empties and end in empties:
        mesh_data = bpy.data.meshes.new(f"bone_{start}_{end}")
        mesh_obj = bpy.data.objects.new(f"Bone_{start}_{end}", mesh_data)
        bpy.context.collection.objects.link(mesh_obj)
        verts = [empties[start].location, empties[end].location]
        edges = [(0, 1)]
        mesh_data.from_pydata(verts, edges, [])
        mesh_data.update()

# === 4. Импорт GLB модели ===
bpy.ops.import_scene.gltf(filepath="/Users/xd/Desktop/nsu_2course/2_sem/AI_PROJ/egor2_mesh.glb")

# === 5. Поиск модели (MESH объект) ===
model = None
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        model = obj
        break

if not model:
    print("GLB-модель не найдена!")
else:
    # === 6. Получаем BoundingBox модели ===
    def get_bbox_center_and_size(obj):
        local_coords = [v.co for v in obj.data.vertices]
        world_coords = [obj.matrix_world @ v for v in local_coords]
        min_corner = mathutils.Vector(map(min, zip(*world_coords)))
        max_corner = mathutils.Vector(map(max, zip(*world_coords)))
        center = (min_corner + max_corner) / 2
        size = max_corner - min_corner
        return center, size

    model_center, model_size = get_bbox_center_and_size(model)

    # === 7. BoundingBox скелета ===
    skel_coords = [e.location for e in empties.values()]
    skel_min = mathutils.Vector(map(min, zip(*skel_coords)))
    skel_max = mathutils.Vector(map(max, zip(*skel_coords)))
    skel_center = (skel_min + skel_max) / 2
    skel_size = skel_max - skel_min

    # === 8. Масштабируем модель под скелет по росту ===
    real_height_m = 1.92
    scale_factor = skel_size.z / real_height_m
    model.scale = [s * scale_factor for s in model.scale]

    bpy.context.view_layer.update()

    # === 9. Пересчитываем центр после масштабирования ===
    model_center_scaled, _ = get_bbox_center_and_size(model)

    # === 10. Центрируем модель по скелету ===
    offset = skel_center - model_center_scaled
    model.location += offset

    bpy.context.view_layer.update()
    print("Модель успешно масштабирована и центрирована.")
