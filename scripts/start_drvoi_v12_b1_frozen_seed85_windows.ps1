$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProject = "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
$runDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\mrat-b1-frozen-seed85"
$analysisDir = Join-Path $projectRoot "docs\analysis\drvoi-rhppo-v12-b1-frozen-seed85"

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
    (Join-Path $analysisDir "summary.json")
)
if (($required | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
    Write-Output "V12-Frozen B1 seed85 training, evaluations, and analysis already completed"
    exit 0
}

$command = "cd '$wslProject' && exec /usr/bin/bash scripts/supervise_drvoi_v12_b1_frozen_seed85_wsl.sh"
$process = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $command) `
    -WindowStyle Hidden -PassThru
Write-Output "started detached V12-Frozen B1 WSL supervisor windows_pid=$($process.Id)"
