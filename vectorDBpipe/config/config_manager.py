import yaml
import os

class ConfigManager:
    """
    Loads and manages configuration settings from config.yaml
    Provides easy attribute-style access to nested keys.
    """

    def __init__(self, config_path: str = "vectorDBpipe/config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        """
        Retrieve value using dot-separated key path, e.g. 'model.name'
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
    print(config.get("vectordb.backend"))
    print(config.get("model.name"))
