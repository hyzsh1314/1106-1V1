$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProject = "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
$dataDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\reachability-a1-seed85"
$modelDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\reachability-model-a2-seed85"
$b0Dir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\mrat-b0-seed85"

if ((Test-Path (Join-Path $dataDir "COMPLETED.json")) -and
    (Test-Path (Join-Path $modelDir "COMPLETED.json")) -and
    (Test-Path (Join-Path $b0Dir "COMPLETED.json")) -and
    (Test-Path (Join-Path $b0Dir "eval-hold.json")) -and
    (Test-Path (Join-Path $b0Dir "eval-pursuit.json")) -and
    (Test-Path (Join-Path $b0Dir "eval-evasive.json"))) {
    Write-Output "V12 reachability A1/A2 and MRAT B0 already completed"
    exit 0
}

$command = "cd '$wslProject' && exec /usr/bin/bash scripts/supervise_drvoi_v12_reachability_wsl.sh"
$process = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $command) `
    -WindowStyle Hidden -PassThru
Write-Output "started detached WSL supervisor windows_pid=$($process.Id)"
