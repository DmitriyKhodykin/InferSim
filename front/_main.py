import sys, subprocess
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import os

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.utils import (
    parse_infersim_output,
    estimate_max_parallel,
    get_gpu_options,
    get_model_options,
    get_model_by_name
)

st.set_page_config(layout="wide", page_title="InferSim Dashboard")
st.title("InferSim – Моделирование инфереса LLM на GPU")

# ---------- Боковая панель ----------
with st.sidebar:
    st.header("Параметры")

    # Модель
    available_models = get_model_options()
    selected_model_name = st.selectbox("Модель", available_models, index=0)
    model_info = get_model_by_name(selected_model_name)  # dict с параметрами

    # GPU
    gpu_list = get_gpu_options()
    gpu_names = [g["name"] for g in gpu_list]
    selected_gpu_names = st.multiselect("Выберите GPU", gpu_names, default=["H200"])
    selected_gpus = {g["name"]: g for g in gpu_list if g["name"] in selected_gpu_names}

    # Диапазоны токенов
    input_bins = [128, 256, 512, 1024, 2048, 4096, 8192]
    output_bins = [128, 256, 512, 1024, 2048, 4096, 8192]

    # RPS
    rps = st.number_input("Целевой RPS", min_value=0.0, value=1.0, step=0.1)

    run_sim = st.button("Запустить симуляцию")

# ---------- Запуск InferSim ----------
def run_infersim(gpu_info, model_info, input_lengths, output_lengths):
    device_type = gpu_info["device_type"]
    world_size = gpu_info["world_size"]
    config_rel = model_info["config_path"]            # например "hf_configs/qwen3_32b_config.json"
    config_path = PROJECT_ROOT / "back" / config_rel

    ttft_matrix, tpot_matrix = [], []
    for out_tok in output_lengths:
        ttft_row, tpot_row = [], []
        for in_tok in input_lengths:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "back" / "main.py"),
                "--config-path", str(config_path),
                "--device-type", device_type,
                "--world-size", str(world_size),
                "--target-isl", str(in_tok),
                "--target-osl", str(out_tok),
            ]

            env = os.environ.copy()
            env['PYTHONPATH'] = str(PROJECT_ROOT)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(PROJECT_ROOT),
                env=env
            )

            ttft, tpot = parse_infersim_output(proc.stdout + proc.stderr)
            ttft_row.append(ttft if ttft is not None else 9999)
            tpot_row.append(tpot if tpot is not None else 9999)
        ttft_matrix.append(ttft_row)
        tpot_matrix.append(tpot_row)

    max_par = estimate_max_parallel(gpu_info["name"], model_info["name"], 4096, 256)
    return {
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "ttft_ms": ttft_matrix,
        "tpot_ms": tpot_matrix,
        "max_parallel_requests": max_par,
    }

# ---------- Отображение ----------
if run_sim:
    results = {}
    with st.spinner("Выполняется симуляция..."):
        for gpu_name, cfg in selected_gpus.items():
            data = run_infersim(cfg, model_info, input_bins, output_bins)
            if data:
                results[gpu_name] = data

    if not results:
        st.warning("Нет данных для отображения")
        st.stop()

    tabs = st.tabs(list(results.keys()))
    for tab, (gpu_name, data) in zip(tabs, results.items()):
        with tab:
            # ... (визуализация остаётся без изменений) ...
            st.subheader(f"Тепловые карты задержек – {gpu_name} (модель: {selected_model_name})")
            x_vals = data["input_lengths"]
            y_vals = data["output_lengths"]
            ttft = data["ttft_ms"]
            tpot = data["tpot_ms"]
            max_par = data["max_parallel_requests"]

            prefill_sec = [[ttft[i][j]/1000 for j in range(len(x_vals))] for i in range(len(y_vals))]
            decode_sec = [[(tpot[i][j]/1000)*y_vals[i] for j in range(len(x_vals))] for i in range(len(y_vals))]
            e2e_sec = [[prefill_sec[i][j] + decode_sec[i][j] for j in range(len(x_vals))] for i in range(len(y_vals))]

            def heatmap(title, values):
                fig = go.Figure(data=go.Heatmap(
                    x=x_vals, y=y_vals, z=values,
                    colorscale='Viridis',
                    # zmin=1, zmax=30,
                    colorbar=dict(title="Секунды")
                ))
                fig.update_layout(title=title,
                                  xaxis_title="Входные токены",
                                  yaxis_title="Выходные токены",
                                  xaxis_type="log", yaxis_type="log")
                return fig

            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(heatmap("Prefill (сек)", prefill_sec), use_container_width=True)
            with col2:
                st.plotly_chart(heatmap("Decode (сек)", decode_sec), use_container_width=True)
            with col3:
                st.plotly_chart(heatmap("E2E Total (сек)", e2e_sec), use_container_width=True)

            if rps > 0:
                try:
                    idx_in = x_vals.index(4096)
                    idx_out = y_vals.index(256)
                except ValueError:
                    idx_in = idx_out = 0
                e2e_ref = e2e_sec[idx_out][idx_in]
                req_par = rps * e2e_ref
                st.write(f"**Референс (вх=4096, вых=256):** E2E={e2e_ref:.2f}с → требуется {req_par:.1f} параллельных запросов, макс. по памяти={max_par}")
                if req_par > max_par:
                    st.error("⚠️ Памяти недостаточно!")
                else:
                    st.success("✅ Памяти достаточно.")