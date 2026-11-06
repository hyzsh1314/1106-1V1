# Upstream and provenance

AC-COSR-WVR is a research-method layer built on the imported legacy LAG
environment. The legacy source tree remains at the repository root to avoid a
large path-only migration and to preserve existing checkpoints and artifacts.

| Item | Value |
| --- | --- |
| Upstream project | [liuqh16/LAG](https://github.com/liuqh16/LAG) |
| Upstream reference recorded by the design specification | `master@ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08` |
| Imported repository baseline | `main@e1843aad80747edb962c501f58ba2cc0d0f25b81` |
| JSBSim source declaration | `envs/JSBSim/data/.gitmodules` names `https://github.com/JSBSim-Team/jsbsim.git` |
| JSBSim revision in this working tree | Unknown: the directory is vendored and has no `.git` metadata; no SHA is invented. |
| Licence | Existing GPL-2.0 `LICENSE` remains authoritative. |

COSR-specific work is isolated under `ac_cosr_wvr/`, `configs/cosr/`,
`scripts/cosr/`, `tests/cosr/`, and `docs/`. The small compatibility changes in
the legacy layer are listed in `docs/audit/CODE_REVIEW_HANDOFF.md`.
