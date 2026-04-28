import re
import math
from pathlib import Path
import yaml


# ---------- парсинг вывода InferSim ----------
def parse_infersim_output(text: str):
    ttft_ms = tpot_ms = None
    for line in text.splitlines():
        if "TTFT (ms):" in line:
            try:
                ttft_ms = float(line.split(":")[1].strip())
            except:
                pass
        if "TPOT (ms):" in line:
            try:
                tpot_ms = float(line.split(":")[1].strip())
            except:
                pass
    return ttft_ms, tpot_ms


# ---------- загрузка конфигураций ----------
def load_gpu_config():
    config_path = Path(__file__).parent / "gpu_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f).get("gpus", [])


def load_model_config():
    config_path = Path(__file__).parent / "model_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f).get("models", [])


def get_gpu_by_name(name: str):
    for gpu in load_gpu_config():
        if gpu["name"] == name:
            return gpu
    raise ValueError(f"GPU '{name}' не найден")


def get_model_by_name(name: str):
    for model in load_model_config():
        if model["name"] == name:
            return model
    raise ValueError(f"Модель '{name}' не найдена")


def get_gpu_options():
    return [
        {
            "name": g["name"],
            "device_type": g["device_type"],
            "world_size": g["world_size"],
            "display_name": g.get("display_name", g["name"]),
        }
        for g in load_gpu_config()
    ]


def get_model_options():
    return [m["name"] for m in load_model_config()]


# ---------- оценка максимальной параллельности ----------
def estimate_max_parallel(
    gpu_name: str, model_name: str, input_tokens: int, output_tokens: int
) -> int:
    gpu = get_gpu_by_name(gpu_name)
    model = get_model_by_name(model_name)

    memory_gb = gpu["memory_gb"]
    avail_gb = memory_gb - model["weight_gb"] - model["overhead_gb"]

    bytes_per_token = (
        2
        * model["num_layers"]
        * model["num_kv_heads"]
        * model["head_dim"]
        * model["dtype_bytes"]
    )
    kv_gb_per_req = (bytes_per_token * (input_tokens + output_tokens)) / (1024**3)
    if kv_gb_per_req <= 0:
        return 9999
    return math.floor(avail_gb / kv_gb_per_req)


def compute_memory_usage(gpu_name: str, model_name: str, input_tokens: int,
                         output_tokens: int, concurrency: int) -> dict:
    """
    Возвращает словарь:
      - total_gpu_mem:   общий объём памяти GPU (ГБ)
      - static_mem:      память под веса + оверхед (ГБ)
      - kv_cache_per_req: KV‑кеш на один запрос (ГБ)
      - used_kv:         KV‑кеш на все параллельные запросы (ГБ)
      - total_used:      общая занятая память (ГБ)
    """
    gpu = get_gpu_by_name(gpu_name)
    model = get_model_by_name(model_name)

    total_gpu_mem = gpu["memory_gb"]
    static_mem = model["weight_gb"] + model["overhead_gb"]

    bytes_per_token = (
        2
        * model["num_layers"]
        * model["num_kv_heads"]
        * model["head_dim"]
        * model["dtype_bytes"]
    )
    kv_cache_per_req = bytes_per_token * (input_tokens + output_tokens) / (1024 ** 3)
    used_kv = kv_cache_per_req * concurrency
    total_used = static_mem + used_kv

    return {
        "total_gpu_mem": total_gpu_mem,
        "static_mem": static_mem,
        "kv_cache_per_req": kv_cache_per_req,
        "used_kv": used_kv,
        "total_used": total_used,
    }


def analyze_rps_feasibility(
    gpu_name: str,
    model_name: str,
    ref_in: int,
    ref_out: int,
    rps: float,
    e2e_ref: float,
    max_par: int,
) -> dict:
    """
    Возвращает словарь:
      - required_par:   требуемое число параллельных запросов
      - feasible:       True, если памяти хватает
      - memory_info:    результат compute_memory_usage (или None, если feasible=False)
    """
    required_par = rps * e2e_ref
    feasible = required_par <= max_par
    mem_info = None
    if feasible:
        mem_info = compute_memory_usage(
            gpu_name, model_name, ref_in, ref_out, required_par
        )
    return {
        "required_par": required_par,
        "feasible": feasible,
        "memory_info": mem_info,
    }
