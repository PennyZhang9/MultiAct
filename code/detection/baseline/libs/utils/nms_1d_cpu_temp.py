from torch.utils.cpp_extension import load
import os

# 自动从 CONDA_PREFIX 获取路径更安全
conda_bin = os.path.join(os.environ.get("CONDA_PREFIX", "/opt/conda"), "bin")
os.environ["CXX"] = os.path.join(conda_bin, "x86_64-conda-linux-gnu-c++")
os.environ["CC"] = os.path.join(conda_bin, "x86_64-conda-linux-gnu-gcc")

# 加载 C++ 扩展
nms_1d_cpu = load(
    name="nms_1d_cpu",
    sources=[os.path.join(os.path.dirname(__file__), "csrc/nms_cpu.cpp")],
    verbose=True,
    force=True
)