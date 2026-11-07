$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProject = "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
$dataDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\return-teacher-a6-onpolicy-h60-20k-seed85"
$modelDir = Join-Path $projectRoot "runs\drvoi_rhppo_v12\return-model-a7-onpolicy-behavior-h60-20k-seed85"
if ((Test-Path (Join-Path $dataDir "COMPLETED.json")) -and
    (Test-Path (Join-Path $dataDir "validation.json")) -and
    (Test-Path (Join-Path $modelDir "COMPLETED.json"))) {
    Write-Output "V12 on-policy return A6/A7 already completed"
    exit 0
}
$command = "cd '$wslProject' && exec /usr/bin/bash scripts/supervise_drvoi_v12_onpolicy_return_a6_a7_wsl.sh"
$process = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $command) `
    -WindowStyle Hidden -PassThru
Write-Output "started detached V12 on-policy return A6/A7 windows_pid=$($process.Id)"
