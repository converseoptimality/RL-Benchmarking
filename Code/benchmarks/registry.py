"""
Simple registry that loads YAML specs from benchmarks/specs and registers Gymnasium envs.
"""
from pathlib import Path
import yaml
from gymnasium.envs.registration import register

_SPEC_ROOT = Path(__file__).parent / "specs"
_REGISTRY = {}

def load_spec(spec_id: str) -> dict:
    fn = _SPEC_ROOT / f"{spec_id.replace('/', '_')}.yaml"
    with open(fn, "r") as f:
        return yaml.safe_load(f)

def register_all():
    from .envs import make_from_spec  # dispatch to family-specific makers
    for yml in _SPEC_ROOT.glob("*.yaml"):
        with open(yml, "r") as f:
            spec = yaml.safe_load(f)
        gym_id = spec["id"].replace("/", "-")
        _REGISTRY[gym_id] = spec
        family = spec["family"].lower()
        register(
            id=f"{gym_id}-v0",
            entry_point="benchmarks.envs:make_from_spec",
            kwargs={"spec": spec},
            max_episode_steps=spec["horizon"],
        )

def get_spec(gym_id: str) -> dict:
    return _REGISTRY[gym_id]
