# Настройка окружения tailor_env

был создан файл setting_env.ipynb для настройки conda-окружения для работы Tailor-net.

---

## Что устанавливается

- `PyTorch` (CUDA 11.8)
- `scipy`, `numpy`
- `fork chumpy`
- `psbody.mesh`— (для работы с 3D-мешами от MPI-IS)

---

## Требования

- активировано окружение Conda с именем `tailor_env`
- установлен `git`
- есть поддержка `GPU` и `CUDA 11.8+`


---

##  Установка

### 1. Установка PyTorch с поддержкой CUDA 11.8

```bash
/opt/conda/envs/tailor_env/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Установка scipy

```bash
/opt/conda/envs/tailor_env/bin/python -m pip install scipy
```

### 3. Клонирование и установка форка chumpy

```bash
git clone https://github.com/wangsen1312/chumpy.git
cd chumpy
/opt/conda/envs/tailor_env/bin/python -m pip install .
cd ..
```

### 4. Проверка версии numpy

```python
import numpy
print(numpy.__version__)
```

### 5. Установка psbody.mesh

```bash
git clone https://github.com/MPI-IS/mesh.git
cd mesh
git checkout v0.4
/opt/conda/envs/tailor_env/bin/python -m pip install .
cd ..
```

---

## Проверка установки

```python
import torch
import scipy
import chumpy
from psbody.mesh import Mesh

print("Torch:", torch.__version__)
print("Scipy:", scipy.__version__)
print("Chumpy:", chumpy.__version__)
```

---

