$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProject = "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
$runDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\mrat-b6-onpolicy-gated-smoke-seed85"
$analysis = Join-Path $projectRoot "docs\analysis\drvoi-rhppo-v12-b6-onpolicy-gated-smoke\summary.json"
$required = @(
    (Join-Path $runDir "COMPLETED.json"),
    (Join-Path $runDir "eval-hold.json"),
    (Join-Path $runDir "eval-hold-episodes.jsonl"),
    (Join-Path $runDir "eval-hold-segments.jsonl"),
    (Join-Path $runDir "eval-pursuit.json"),
    (Join-Path $runDir "eval-pursuit-episodes.jsonl"),
    (Join-Path $runDir "eval-pursuit-segments.jsonl"),
    (Join-Path $runDir "eval-evasive.json"),
    (Join-Path $runDir "eval-evasive-episodes.jsonl"),
    (Join-Path $runDir "eval-evasive-segments.jsonl"),
    $analysis
)
if (($required | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
    Write-Output "V12 B6 on-policy-gated smoke already completed"
    exit 0
}
$command = "cd '$wslProject' && exec /usr/bin/bash scripts/supervise_drvoi_v12_b6_onpolicy_gated_smoke_wsl.sh"
$process = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $command) `
    -WindowStyle Hidden -PassThru
Write-Output "started detached V12 B6 on-policy-gated smoke windows_pid=$($process.Id)"
