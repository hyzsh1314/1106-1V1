"""Immutable scientific, implementation, and formal-run identities."""
from __future__ import annotations
import json, platform, re, subprocess, sys
try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 compatibility boundary
    import importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from .schemas import ImplementationManifest, RunPlanManifest, ScientificProtocolManifest, canonical_json_bytes, sha256_bytes, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError): return "unavailable"

def _files(*roots):
    result = []
    for root in roots:
        path = PROJECT_ROOT / root
        if path.is_file(): result.append(path)
        elif path.is_dir(): result.extend(x for x in path.rglob("*") if x.is_file() and ".git" not in x.parts)
        else: raise RuntimeError("implementation bundle source missing: {0}".format(path))
    return sorted(set(result), key=lambda x: str(x.relative_to(PROJECT_ROOT)).replace("\\", "/"))

def bundle_sha256(*roots):
    return sha256_bytes(canonical_json_bytes([{"path": str(x.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": sha256_file(x)} for x in _files(*roots)]))

def _upstream_provenance():
    text = (PROJECT_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    upstream, baseline = re.search(r"master@([0-9a-f]{40})", text), re.search(r"main@([0-9a-f]{40})", text)
    if not upstream or not baseline: raise RuntimeError("UPSTREAM.md lacks immutable upstream/import provenance")
    return upstream.group(1), baseline.group(1)

def _jsbsim_identity():
    try:
        import jsbsim
        return "jsbsim:{0}".format(getattr(jsbsim, "__version__", "unknown"))
    except ImportError: return "jsbsim:unavailable"

def _distribution_version(name):
    """Return an exact installed distribution version without importing it."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"

def implementation_environment_fingerprint():
    """All runtime facts that make a formal implementation executable."""
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "torch": _distribution_version("torch"),
        "gymnasium": _distribution_version("gymnasium"),
        "jsbsim": _distribution_version("jsbsim"),
        "pymap3d": _distribution_version("pymap3d"),
        "pandas": _distribution_version("pandas"),
        "pyarrow": _distribution_version("pyarrow"),
        "PyYAML": _distribution_version("PyYAML"),
    }
    cuda = {"enabled": bool(torch_cuda_available())}
    if cuda["enabled"]:
        import torch
        cuda.update({"cuda": str(torch.version.cuda), "cudnn": str(torch.backends.cudnn.version()),
                     "gpu_model": str(torch.cuda.get_device_name(0)), "driver": _nvidia_driver_version()})
    return {"versions": versions, "cuda": cuda}

def torch_cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False

def _nvidia_driver_version():
    try:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], universal_newlines=True)
        return value.splitlines()[0].strip() if value.strip() else "unavailable"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"

def build_implementation_manifest():
    upstream, baseline = _upstream_provenance()
    bundles = {
      "lag_environment_bundle_sha256": bundle_sha256("envs/JSBSim/core", "envs/JSBSim/envs", "envs/JSBSim/tasks", "envs/JSBSim/reward_functions", "envs/JSBSim/termination_conditions", "envs/JSBSim/utils", "envs/JSBSim/configs/1v1/NoWeapon/Selfplay.yaml", "envs/JSBSim/data/aircraft/f16", "envs/JSBSim/data/engine/f16_engine.xml", "envs/JSBSim/data/engine/f16_thruster.xml", "envs/JSBSim/model/baseline.py", "envs/JSBSim/model/baseline_actor.py", "envs/JSBSim/model/baseline_model.pt"),
      "cosr_environment_bundle_sha256": bundle_sha256("ac_cosr_wvr/env", "ac_cosr_wvr/protocol.py", "ac_cosr_wvr/schemas.py", "ac_cosr_wvr/policies/action_mapping.py"),
      "algorithm_bundle_sha256": bundle_sha256("ac_cosr_wvr/algorithms", "ac_cosr_wvr/training"),
      "data_pipeline_bundle_sha256": bundle_sha256("ac_cosr_wvr/data", "ac_cosr_wvr/evaluation")}
    fingerprint = implementation_environment_fingerprint()
    payload = {"repository_commit": _git_sha(), "upstream_lag_commit": upstream, "legacy_import_baseline": baseline, "jsbsim_identity": _jsbsim_identity(), "python_environment_identity": "{0}|{1}".format(sys.version, platform.platform()), "environment_fingerprint": fingerprint, **bundles}
    return ImplementationManifest(**payload, implementation_sha256=sha256_bytes(canonical_json_bytes(payload)))

def build_scientific_protocol_manifest(config):
    config = dict(config); scenario = dict(config.get("scenario", {}))
    missing = [x for x in ("protocol_id", "protocol_version", "protocol_status") if config.get(x) is None]
    if missing: raise RuntimeError("scientific protocol fields missing: {0}".format(", ".join(missing)))
    clean = json.loads(json.dumps(config))
    for key in ("protocol_sha256", "scientific_protocol_sha256", "implementation_sha256", "run_plan_sha256"): clean.pop(key, None)
    legacy_path = Path(config.get("legacy_task_config_path", "envs/JSBSim/configs/1v1/NoWeapon/Selfplay.yaml")); legacy_path = legacy_path if legacy_path.is_absolute() else PROJECT_ROOT / legacy_path
    if not legacy_path.is_file(): raise RuntimeError("legacy task config is missing")
    payload = {"protocol_id": str(config["protocol_id"]), "protocol_version": str(config["protocol_version"]), "scientific_protocol": bool(config.get("scientific_protocol")), "protocol_status": str(config["protocol_status"]), "scientific_config_sha256": sha256_bytes(canonical_json_bytes(clean)), "scenario_semantics": scenario, "autogun_semantics": dict(config.get("autogun", {})), "safety_semantics": dict(config.get("safety", {})), "reward_semantics": {"training_reward": dict(config.get("training_reward", {})), "legacy_task_config_sha256": sha256_file(legacy_path), "legacy_reward_components": list(config.get("legacy_reward_components", ["AltitudeReward", "PostureReward", "EventDrivenReward"]))}, "split_semantics": dict(config.get("splits", {})), "evaluation_semantics": dict(config.get("evaluation", {}))}
    digest = sha256_bytes(canonical_json_bytes(payload)); declared = config.get("protocol_sha256") or config.get("scientific_protocol_sha256")
    if declared is not None and str(declared) != digest: raise RuntimeError("declared scientific protocol SHA mismatch")
    return ScientificProtocolManifest(**payload, scientific_protocol_sha256=digest)

def build_protocol_manifest(config, lag_config_name="1v1/NoWeapon/Selfplay"): return build_scientific_protocol_manifest(config)

def build_run_plan_manifest(config, algorithm, training_seed, total_env_steps, n_envs, learning_rate, hidden_size, checkpoint_interval, eval_interval, opponent, split_code=0, output_schema_version=1):
    algorithm = str(algorithm)
    if algorithm not in ("ppo", "sac", "td3"): raise ValueError("unknown algorithm")
    if isinstance(n_envs, bool) or int(n_envs) <= 0: raise ValueError("n_envs must be a positive integer")
    n_envs = int(n_envs)
    training = dict(config.get("training", {}))
    names = {"ppo": ("gamma", "gae_lambda", "ppo_rollout_steps", "ppo_epochs", "ppo_minibatch_size", "ppo_clip_ratio", "ppo_value_coefficient", "ppo_entropy_coefficient", "ppo_max_grad_norm"), "sac": ("gamma", "batch_size", "replay_capacity", "warmup_steps", "update_after", "updates_per_step", "sac_tau", "sac_target_entropy"), "td3": ("gamma", "batch_size", "replay_capacity", "warmup_steps", "update_after", "updates_per_step", "td3_tau", "td3_policy_delay", "td3_target_noise", "td3_noise_clip", "td3_exploration_noise")}[algorithm]
    missing = [x for x in names if training.get(x) is None]
    if missing: raise RuntimeError("unresolved {0} run-plan hyperparameters: {1}".format(algorithm, ", ".join(missing)))
    opponent = dict(opponent); required = ("opponent_id", "opponent_lineage_id", "opponent_manifest_sha256", "opponent_checkpoint_sha256", "opponent_code", "registry_file_sha256")
    missing = [x for x in required if opponent.get(x) is None]
    if missing: raise RuntimeError("run plan opponent fields missing: {0}".format(", ".join(missing)))
    payload = {"run_plan_id": "{0}-{1}-seed{2}".format(config.get("protocol_id"), algorithm, int(training_seed)), "algorithm": algorithm, "training_seed": int(training_seed), "total_env_steps": int(total_env_steps), "n_envs": n_envs, "learning_rate": float(learning_rate), "hidden_size": int(hidden_size), "checkpoint_interval": int(checkpoint_interval), "eval_interval": int(eval_interval), "algorithm_hyperparameters": {x: training[x] for x in names}, "candidate_role": "candidate", "opponent": opponent, "scenario": {"master_seed": int(training_seed), "split_code": int(split_code), "opponent_code": int(opponent["opponent_code"]), "logical_environment_seed_rule": "Seed(master_seed,run_id,env_id,episode_id,subsystem)"}, "output_schema_version": int(output_schema_version)}
    return RunPlanManifest(**payload, run_plan_sha256=sha256_bytes(canonical_json_bytes(payload)))

def validate_environment_protocol(config, lag_config_name="1v1/NoWeapon/Selfplay"):
    protocol = build_scientific_protocol_manifest(config)
    if protocol.scientific_protocol:
        if protocol.protocol_status != "FROZEN" or not (config.get("protocol_sha256") or config.get("scientific_protocol_sha256")): raise RuntimeError("scientific COSR environment requires a declared frozen scientific protocol")
    elif config.get("smoke_restrictions", {}).get("non_scientific_values") is not True: raise RuntimeError("non-scientific execution requires explicit smoke restrictions")
    return protocol

@dataclass(frozen=True)
class ExecutionAuthorization(object):
    authorization_id: str; scientific_protocol_sha256: str; implementation_sha256: str; run_plan_sha256: str; execution_device: str; authorized: bool
    @classmethod
    def from_path(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8")); names = ("authorization_id", "scientific_protocol_sha256", "implementation_sha256", "run_plan_sha256", "execution_device", "authorized")
        missing = [x for x in names if x not in value]
        if missing: raise RuntimeError("authorization manifest missing: {0}".format(", ".join(missing)))
        return cls(**{x: value[x] for x in names})

def verify_execution_authorization(protocol, implementation, run_plan, authorization, runtime_device):
    if not authorization.authorized: raise RuntimeError("formal execution is not authorized")
    if str(authorization.execution_device) != str(runtime_device): raise RuntimeError("authorization execution_device differs from runtime device")
    if (authorization.scientific_protocol_sha256, authorization.implementation_sha256, authorization.run_plan_sha256) != (protocol.scientific_protocol_sha256, implementation.implementation_sha256, run_plan.run_plan_sha256): raise RuntimeError("authorization does not bind the selected protocol, implementation, and run plan")
    return True
