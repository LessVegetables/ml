Этот скрипт для Blender автоматизирует процесс совмещения 3D-модели человека в формате GLB с позой, полученной из OpenPose, на основе ключевых точек. Он написан на Python с использованием Blender API (bpy) и работает в среде Blender. Цель — отобразить позу человека в 3D-пространстве и масштабировать/центрировать загруженную модель под неё, сохранив пропорции и позиционирование.

На входе скрипт принимает:
    1) JSON-файл с позой в формате OpenPose (pose_keypoints_2d)
    2) GLB-файл с 3D-моделью человека
    3) Рост человека в метрах (здесь установлен - 1.92), используется для правильного масштабирования.

Принцип работы скрипта:
    1) Загружается JSON с ключевыми точками OpenPose. Из массива координат отфильтровываются только те точки, которые имеют уверенность (confidence) больше 0.5. Точки масштабируются и преобразуются в 3D-формат с размещением по X и Z осям (ось Y используется как нулевая плоскость).
    2) Для каждой валидной точки создается Empty-объект в сцене, отображающий положение сустава.
    3) Эмпти соединяются в "скелет" на основе стандартной схемы COCO (список пар индексов суставов).
    4) Загружается GLB-модель человека.
    5) Скрипт находит MESH-объект модели среди загруженных.
    6) Определяется bounding box (границы) модели и вычисляется её центр и размер.
    7) Аналогично вычисляется bounding box всего скелета.
    8) Рассчитывается масштаб модели так, чтобы её рост совпадал с высотой скелета, соответствующей реальному росту человека.
    9) Центр модели пересчитывается после масштабирования.
    10) Модель сдвигается в пространстве так, чтобы её центр совпадал с центром скелета.

Результат выполнения:
    1) В сцене Blender визуализируется поза человека, полученная из OpenPose.
    2) 3D-модель корректно масштабируется и совмещается с этой позой по высоте и положению.
    3) Полученная сцена может и планируется использоваться для измерений частей тела(например, обхвата, длины).