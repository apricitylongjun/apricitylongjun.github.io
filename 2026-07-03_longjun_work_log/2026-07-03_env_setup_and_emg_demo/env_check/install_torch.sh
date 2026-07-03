#!/usr/bin/env bash
# 修复 venv 缺 pip，再装支持 RTX 5080(Blackwell sm_120) 的 PyTorch
# 无需 sudo；清华源
export PATH="$HOME/.local/bin:$PATH"
VENV_PY="$HOME/aa_workspace/.venv/bin/python"

echo "==[$(date)] 1. ensurepip 补 pip=="
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  "$VENV_PY" -m ensurepip --upgrade 2>&1 | tail -3 || {
    echo "ensurepip 失败，改用 get-pip.py";
    cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py && "$VENV_PY" get-pip.py -i https://pypi.tuna.tsinghua.edu.cn/simple;
  }
fi
"$VENV_PY" -m pip --version

echo "==[$(date)] 2. 升级 pip=="
"$VENV_PY" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==[$(date)] 3. 装 torch torchvision (清华源)=="
"$VENV_PY" -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==[$(date)] 4. 验证 torch + Blackwell 支持=="
"$VENV_PY" - <<'PY'
import torch
print("torch:", torch.__version__, "| 编译CUDA:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
    al = torch.cuda.get_arch_list()
    print("arch_list:", al)
    print("支持Blackwell sm_120:", any(str(x).replace('sm_','').startswith('12') for x in al))
PY
echo "==[$(date)] DONE=="
