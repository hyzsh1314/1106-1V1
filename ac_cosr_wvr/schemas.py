"""Stable, serialisable schemas shared by COSR components.

All new manifests use canonical split names.  The schemas deliberately keep
scientific identity (lineage, hashes, and split) separate from runtime data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


CANONICAL_SPLITS = (
    "policy_train", "id_train", "cal", "matrix", "validation", "final_test"
)
_SPLIT_ALIASES = {"val": "validation", "test": "final_test"}


def canonical_split(value):
    """Return a canonical split name, accepting legacy aliases at boundaries."""
    value = _SPLIT_ALIASES.get(value, value)
    if value not in CANONICAL_SPLITS:
        raise ValueError("unknown split: {0}".format(value))
    return value


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class ManifestMixin(object):
    def to_dict(self):
        return asdict(self)

    def sha256(self):
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class ProtocolManifest(ManifestMixin):
    """Immutable identity for a frozen COSR execution protocol.

    ``protocol_sha256`` is computed from the protocol payload, including the
    legacy task inputs that still supply observations and dense rewards.  It
    deliberately does not encode an execution permission: authorization is a
    separate operational document owned by the launcher.
    """

    protocol_id: str
    protocol_version: str
    scientific_protocol: bool
    protocol_status: str
    protocol_sha256: str
    code_commit: str
    lag_base_commit: str
    legacy_task_config_path: str
    legacy_task_sha256: str
    legacy_task_config_sha256: str
    environment_sha256: str
    jsbsim_source_identity: str
    control_interval_s: float
    reward_components: Dict[str, Any]
    config_sha256: str

    def __post_init__(self):
        if not all((self.protocol_id, self.protocol_version, self.protocol_sha256,
                    self.lag_base_commit, self.legacy_task_config_path, self.legacy_task_sha256,
                    self.legacy_task_config_sha256,
                    self.environment_sha256, self.jsbsim_source_identity,
                    self.config_sha256)):
            raise ValueError("ProtocolManifest has empty identity fields")
        if self.protocol_status not in ("SMOKE", "DRAFT", "FROZEN"):
            raise ValueError("invalid protocol status: {0}".format(self.protocol_status))


@dataclass(frozen=True)
class ScientificProtocolManifest(ManifestMixin):
    protocol_id: str
    protocol_version: str
    scientific_protocol: bool
    protocol_status: str
    scientific_config_sha256: str
    scientific_protocol_sha256: str
    scenario_semantics: Dict[str, Any]
    autogun_semantics: Dict[str, Any]
    safety_semantics: Dict[str, Any]
    reward_semantics: Dict[str, Any]
    split_semantics: Dict[str, Any]
    evaluation_semantics: Dict[str, Any]

    @property
    def protocol_sha256(self):
        """Compatibility spelling; new code should use the explicit name."""
        return self.scientific_protocol_sha256


@dataclass(frozen=True)
class ImplementationManifest(ManifestMixin):
    repository_commit: str
    upstream_lag_commit: str
    legacy_import_baseline: str
    jsbsim_identity: str
    python_environment_identity: str
    environment_fingerprint: Dict[str, Any]
    lag_environment_bundle_sha256: str
    cosr_environment_bundle_sha256: str
    algorithm_bundle_sha256: str
    data_pipeline_bundle_sha256: str
    implementation_sha256: str


@dataclass(frozen=True)
class RunPlanManifest(ManifestMixin):
    run_plan_id: str
    algorithm: str
    training_seed: int
    total_env_steps: int
    n_envs: int
    learning_rate: float
    hidden_size: int
    checkpoint_interval: int
    eval_interval: int
    algorithm_hyperparameters: Dict[str, Any]
    candidate_role: str
    opponent: Dict[str, Any]
    scenario: Dict[str, Any]
    output_schema_version: int
    run_plan_sha256: str


@dataclass(frozen=True)
class ScenarioSpec(ManifestMixin):
    """One replayable initial condition.

    ``red_init``/``blue_init`` use canonical north/east metres and
    ``altitude_m`` as height above mean sea level (h_MSL), never local up.
    """
    schema_version: int
    episode_id: str
    episode_seed: int
    split: str
    scenario_id: str
    stratum: str
    side_swap: bool
    red_init: Dict[str, Any]
    blue_init: Dict[str, Any]
    # ``stratum`` is always stated from the candidate/ego perspective.  The
    # physical red/blue assignment is deliberately a separate concern so a
    # side swap cannot silently reverse "ego advantage" into a disadvantage.
    candidate_side: str = "red"
    dynamics: Dict[str, Any] = field(default_factory=dict)
    sensor: Dict[str, Any] = field(default_factory=dict)
    task_hash: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "split", canonical_split(self.split))
        if self.schema_version < 1 or not self.episode_id:
            raise ValueError("ScenarioSpec requires a schema version and episode id")
        if not self.stratum:
            raise ValueError("ScenarioSpec requires a stratum")
        if self.candidate_side not in ("red", "blue"):
            raise ValueError("candidate_side must be red or blue")


@dataclass(frozen=True)
class PolicyManifest(ManifestMixin):
    policy_id: str
    family: str
    lineage_id: str
    algorithm: str
    observation_interface: str
    action_interface: str
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    config_path: Optional[str]
    config_sha256: Optional[str]
    train_env_steps: int
    seed: int
    environment_commit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        required = (self.policy_id, self.family, self.lineage_id,
                    self.algorithm, self.observation_interface,
                    self.action_interface, self.environment_commit)
        if not all(required):
            raise ValueError("PolicyManifest has empty required identity fields")
        if self.train_env_steps < 0:
            raise ValueError("train_env_steps must be non-negative")
        if bool(self.checkpoint_path) != bool(self.checkpoint_sha256):
            raise ValueError("checkpoint path and hash must be provided together")


@dataclass(frozen=True)
class OpponentManifest(ManifestMixin):
    opponent_id: str
    family: str
    lineage_id: str
    split: str
    ood_level: str
    policy_manifest_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "split", canonical_split(self.split))
        if self.ood_level not in ("id", "near_ood", "far_ood", "extreme"):
            raise ValueError("invalid ood level: {0}".format(self.ood_level))
        if not all((self.opponent_id, self.family, self.lineage_id)):
            raise ValueError("OpponentManifest requires identity fields")


@dataclass(frozen=True)
class MatrixManifest(ManifestMixin):
    matrix_id: str
    split: str
    policy_ids: List[str]
    opponent_ids: List[str]
    source_manifest_sha256: str
    config_sha256: str
    environment_commit: str

    def __post_init__(self):
        object.__setattr__(self, "split", canonical_split(self.split))
        if self.split != "matrix":
            raise ValueError("MatrixManifest must be created from the matrix split")
        if not self.policy_ids or not self.opponent_ids:
            raise ValueError("matrix manifests need policies and opponents")
