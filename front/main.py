import sys
import subprocess
import os
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

# ---------- Настройка путей ----------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utils import (
    analyze_rps_feasibility,
    parse_infersim_output,
    estimate_max_parallel,
    get_gpu_options,
    get_model_options,
    get_model_by_name,
    get_gpu_by_name,
)

# ---------- Конфигурация страницы ----------
st.set_page_config(layout="wide", page_title="InferSim Dashboard")
st.title("InferSim – Моделирование инференса LLM на GPU")

# ---------- Боковая панель ----------
with st.sidebar:
    st.header("Параметры симуляции")

    # Модель
    available_models = get_model_options()
    selected_model_name = st.selectbox("Модель", available_models, index=0)

    gpu_list = get_gpu_options()
    display_names = [
        g.get("display_name", g["name"]) for g in gpu_list
    ]  # fallback на name, если нет display_name
    name_by_display = {d: g["name"] for d, g in zip(display_names, gpu_list)}

    selected_display_names = st.multiselect(
        "Выберите GPU", display_names, default=display_names[:1]
    )
    selected_gpu_names = [name_by_display[d] for d in selected_display_names]
    selected_gpus = {g["name"]: g for g in gpu_list if g["name"] in selected_gpu_names}

    # Диапазоны токенов
    input_bins = [128, 256, 512, 1024, 2048, 4096, 8192]
    output_bins = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Референсная точка для анализа RPS
    st.subheader("Референсная точка")
    ref_in = st.selectbox("Входные токены", input_bins, index=input_bins.index(4096))
    ref_out = st.selectbox("Выходные токены", output_bins, index=output_bins.index(256))

    # Целевой RPS
    rps = st.number_input(
        "Целевой RPS (запросов/с)", min_value=0.0, value=1.0, step=0.1
    )

    run_sim = st.button("Запустить симуляцию")


# ---------- Запуск симуляции (с кэшированием) ----------
@st.cache_data(show_spinner="Выполняется симуляция...")
def run_simulation_cached(
    gpu_name: str,
    model_name: str,
    input_lengths_tuple: tuple,
    output_lengths_tuple: tuple,
):
    """
    Запускает InferSim для каждой комбинации длин и возвращает матрицы задержек.
    Параметры принимаются хэшируемыми, чтобы работал кэш Streamlit.
    """
    gpu_info = get_gpu_by_name(gpu_name)
    model_info = get_model_by_name(model_name)

    device_type = gpu_info["device_type"]
    world_size = gpu_info["world_size"]
    config_path = PROJECT_ROOT / "back" / model_info["config_path"]

    input_lengths = list(input_lengths_tuple)
    output_lengths = list(output_lengths_tuple)

    ttft_matrix = []
    tpot_matrix = []

    for out_tok in output_lengths:
        ttft_row = []
        tpot_row = []
        for in_tok in input_lengths:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "back" / "main.py"),
                "--config-path",
                str(config_path),
                "--device-type",
                device_type,
                "--world-size",
                str(world_size),
                "--target-isl",
                str(in_tok),
                "--target-osl",
                str(out_tok),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(PROJECT_ROOT / "back"),
                env=env,
            )
            ttft, tpot = parse_infersim_output(proc.stdout + proc.stderr)
            ttft_row.append(ttft if ttft is not None else 9999)
            tpot_row.append(tpot if tpot is not None else 9999)
        ttft_matrix.append(ttft_row)
        tpot_matrix.append(tpot_row)

    max_par = estimate_max_parallel(
        gpu_name, model_name, input_lengths[0], output_lengths[0]
    )  # будет пересмотрено ниже
    return {
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "ttft_ms": ttft_matrix,
        "tpot_ms": tpot_matrix,
        "max_parallel_requests": max_par,
    }


# ---------- Отображение результатов ----------
if run_sim:
    results = {}

    for gpu_name in selected_gpu_names:
        data = run_simulation_cached(
            gpu_name, selected_model_name, tuple(input_bins), tuple(output_bins)
        )
        if data:
            # Пересчитываем max_parallel для выбранной референсной точки
            data["max_parallel_requests"] = estimate_max_parallel(
                gpu_name, selected_model_name, ref_in, ref_out
            )
            results[gpu_name] = data

    if not results:
        st.warning("Нет данных для отображения.")
        st.stop()

    gpu_name_to_display = {g["name"]: g.get("display_name", g["name"]) for g in gpu_list}
    tabs = st.tabs([gpu_name_to_display[name] for name in results.keys()])

    for tab, (gpu_name, data) in zip(tabs, results.items()):
        with tab:
            st.subheader(
                f"Тепловые карты задержек – {gpu_name} (модель: {selected_model_name})"
            )

            x_vals = data["input_lengths"]
            y_vals = data["output_lengths"]
            ttft = data["ttft_ms"]
            tpot = data["tpot_ms"]
            max_par = data["max_parallel_requests"]

            prefill_sec = [
                [ttft[i][j] / 1000.0 for j in range(len(x_vals))]
                for i in range(len(y_vals))
            ]
            decode_sec = [
                [(tpot[i][j] / 1000.0) * y_vals[i] for j in range(len(x_vals))]
                for i in range(len(y_vals))
            ]
            e2e_sec = [
                [prefill_sec[i][j] + decode_sec[i][j] for j in range(len(x_vals))]
                for i in range(len(y_vals))
            ]

            vmin, vmax = 0.0, 40.0

            def make_heatmap(title, matrix):
                cell_text = [[f"{val:.1f}" for val in row] for row in matrix]
                fig = go.Figure(
                    data=go.Heatmap(
                        x=x_vals,
                        y=y_vals,
                        z=matrix,
                        text=cell_text,
                        texttemplate="%{text}",
                        textfont=dict(color="white"),
                        colorscale="RdYlGn_r",
                        zmin=vmin,
                        zmax=vmax,
                        colorbar=dict(title="Секунды"),
                    )
                )
                fig.update_layout(
                    title=title,
                    xaxis_title="Входные токены",
                    yaxis_title="Выходные токены",
                    xaxis_type="log",
                    yaxis_type="log",
                    width=400,
                    height=400,
                )
                return fig

            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(
                    make_heatmap("Prefill (сек)", prefill_sec), width="stretch"
                )
            with col2:
                st.plotly_chart(
                    make_heatmap("Decode (сек)", decode_sec), width="stretch"
                )
            with col3:
                st.plotly_chart(
                    make_heatmap("E2E Total (сек)", e2e_sec), width="stretch"
                )

            # Анализ RPS
            # Анализ RPS
            if rps > 0:
                try:
                    idx_in = x_vals.index(ref_in)
                    idx_out = y_vals.index(ref_out)
                except ValueError:
                    idx_in = idx_out = 0
                e2e_ref = e2e_sec[idx_out][idx_in]

                feasibility = analyze_rps_feasibility(
                    gpu_name, selected_model_name, ref_in, ref_out, rps, e2e_ref, max_par
                )

                st.write(
                    f"**Референс (вх={ref_in}, вых={ref_out}):** "
                    f"E2E = {e2e_ref:.2f} с → "
                    f"требуется {feasibility['required_par']:.1f} параллельных запросов, "
                    f"максимум по памяти = {max_par}"
                )

                if feasibility["feasible"]:
                    mem = feasibility["memory_info"]
                    st.success(
                        f"✅ Памяти достаточно. "
                        f"[Занято: {mem['total_used']:.0f} ГБ из {mem['total_gpu_mem']} ГБ]"
                    )
                else:
                    st.error(
                        "⚠️ Памяти недостаточно! Уменьшите RPS или длину контекста."
                    )
