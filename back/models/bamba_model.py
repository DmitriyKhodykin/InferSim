import yaml
from pathlib import Path
from models.hybrid_model import HybridModel


class MambaHybridModel(HybridModel):
    def __init__(self, args, config):
        super().__init__(args, config)

        # Загружаем model_params из model_config.yaml
        cfg_path = (
            Path(__file__).resolve().parent.parent.parent
            / "common"
            / "model_config.yaml"
        )
        self.model_params = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                models_data = yaml.safe_load(f)
                if models_data and "models" in models_data:
                    for m in models_data["models"]:
                        if m.get("config_path") == getattr(
                            config, "_config_path", "hf_configs/bamba_9b_v2_config.json"
                        ):
                            self.model_params = m
                            break
                    if not self.model_params:
                        for m in models_data["models"]:
                            if m.get("name") == "Bamba-9B-v2-FP16":
                                self.model_params = m
                                break

        # Все слои гибридные
        self.attn_layer_indices = list(range(config.num_hidden_layers))
        self.num_full_attn_layers = config.num_hidden_layers
        self.num_mamba_layers = config.num_hidden_layers
        self.d_state = getattr(config, "d_state", 256)
        self.d_conv = getattr(config, "d_conv", 4)
        self.expand = getattr(config, "expand", 2)

    # Методы get_trainable_params и get_num_params оставляем как есть
    def get_trainable_params(self):
        base = super().get_trainable_params()
        ssm_params = self.num_mamba_layers * self.config.hidden_size * self.d_state * 2
        return base + ssm_params

    def get_num_params(self):
        return self.get_trainable_params()
