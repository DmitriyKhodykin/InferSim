import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "back"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.model_config import ModelConfig
from models.bamba_model import MambaHybridModel

config = ModelConfig("back/hf_configs/bamba_9b_v2_config.json")

# Минимальный набор аргументов, необходимый HybridModel
args = argparse.Namespace(
    device_type="H200",
    world_size=1,
    use_fp8_gemm=False,
    use_fp8_kv=False,
    max_prefill_tokens=4096,
    decode_bs=None,
    target_tgs=2560,
    target_tpot=50,
    target_isl=4096,
    target_osl=256,
)

model = MambaHybridModel(args, config)

print("✅ Класс модели:", type(model).__name__)
print("✅ model_params:", model.model_params)
print("✅ MFU prefill:", model.get_mfu('prefill'))
print("✅ MFU decode :", model.get_mfu('decode'))