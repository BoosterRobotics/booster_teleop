
import pathlib
import json
from typing import Dict, Any

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE.resolve().parents[1] / "assets"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config_index() -> Dict[str, Any]:

    public_config_path = IK_CONFIG_ROOT / "config_index_public.json"
    with open(public_config_path, 'r', encoding='utf-8') as f:
        public_config = json.load(f)
    
    selfhost_config_path = IK_CONFIG_ROOT / "config_index_selfhost.json"
    if selfhost_config_path.exists():
        with open(selfhost_config_path, 'r', encoding='utf-8') as f:
            selfhost_config = json.load(f)
        config = deep_merge(public_config, selfhost_config)
    else:
        config = public_config
    
    return config


def resolve_paths(config: Dict[str, Any], base_path: pathlib.Path) -> Dict[str, Any]:
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = resolve_paths(value, base_path)
        elif isinstance(value, str):
            result[key] = base_path / value
        else:
            result[key] = value
    return result

_config = load_config_index()

ROBOT_XML_DICT = {
    key: value.resolve() if isinstance(value, pathlib.Path) else value
    for key, value in resolve_paths(_config.get("robot_xml", {}), ASSET_ROOT).items()
}

_ik_configs_raw = _config.get("ik_configs", {})
IK_CONFIG_DICT = {}
for source_type, robots in _ik_configs_raw.items():
    IK_CONFIG_DICT[source_type] = {}
    for robot_name, config_file in robots.items():

        selfhost_path = IK_CONFIG_ROOT / "selfhost" / config_file
        public_path = IK_CONFIG_ROOT / config_file
        
        if selfhost_path.exists():
            IK_CONFIG_DICT[source_type][robot_name] = selfhost_path
        else:
            IK_CONFIG_DICT[source_type][robot_name] = public_path

ROBOT_BASE_DICT = _config.get("robot_base", {})
VIEWER_CAM_DISTANCE_DICT = _config.get("viewer_cam_distance", {})
