import yaml
import os
import re

class ConfigManager:
    """
    Loads and manages configuration settings from config.yaml
    Provides easy attribute-style access to nested keys and resolves environment variables.
    """

    def __init__(self, config_path: str = "config.yaml", config_override: dict = None):
        # We check both the root and the internal config paths
        if not os.path.exists(config_path):
            fallback_path = "vectorDBpipe/config/config.yaml"
            if os.path.exists(fallback_path):
                config_path = fallback_path
            elif os.path.exists("../config.yaml"):
                config_path = "../config.yaml"
        self.config_path = config_path
        self.config_override = config_override
        self._config = self._load_config()

    def _resolve_env_vars(self, config_dict):
        """Recursively resolve ${ENV_VAR} strings in the config dictionary."""
        env_pattern = re.compile(r'\$\{([^}^{]+)\}')

        for key, value in config_dict.items():
            if isinstance(value, dict):
                config_dict[key] = self._resolve_env_vars(value)
            elif isinstance(value, list):
                resolved_list = []
                for item in value:
                    if isinstance(item, str):
                        match = env_pattern.search(item)
                        if match:
                            env_var = match.group(1)
                            item = item.replace(f'${{{env_var}}}', os.environ.get(env_var, ''))
                    resolved_list.append(item)
                config_dict[key] = resolved_list
            elif isinstance(value, str):
                match = env_pattern.search(value)
                if match:
                    env_var = match.group(1)
                    config_dict[key] = value.replace(f'${{{env_var}}}', os.environ.get(env_var, ''))
        return config_dict

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {} # Return empty config gracefully if not found to allow defaults

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # If everything is nested under 'vectorDBpipe', lift it
        if "vectorDBpipe" in config:
            config = config["vectorDBpipe"]

        # Resolve ${ENV_VAR} patterns
        config = self._resolve_env_vars(config)

        if self.config_override:
            self._deep_update(config, self.config_override)

        return config

    def _deep_update(self, base_dict, override_dict):
        for key, value in override_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get(self, key_path: str, default=None):
        """
        Retrieve value using dot-separated key path, e.g. 'embedding.provider'
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def config(self):
        return self._config

    def __getitem__(self, key):
        return self._config.get(key)

    def __repr__(self):
        return f"ConfigManager(config_path='{self.config_path}')"

# Example use
if __name__ == "__main__":
    config = ConfigManager()
    print(config.get("database.provider"))
    print(config.get("embedding.model_name"))
