# Downloads the referenced YOLOX-s ONNX model only when missing, or when -Force is supplied.
param([switch]$Force)
$ErrorActionPreference = "Stop"
$ModelUrl = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
$TargetDir = Join-Path $PSScriptRoot "..\models"
$TargetFile = Join-Path $TargetDir "vehicle_detector.onnx"

if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir | Out-Null
}

if ((Test-Path $TargetFile) -and -not $Force) {
    $Existing = Get-Item -Path $TargetFile
    if ($Existing.Length -gt 1000000) {
        $Hash = (Get-FileHash -Path $TargetFile -Algorithm SHA256).Hash.ToLower()
        Write-Host "Existing vehicle model found; leaving it unchanged."
        Write-Host "File     : $TargetFile"
        Write-Host "Size     : $($Existing.Length) bytes"
        Write-Host "SHA-256  : $Hash"
        Write-Host "Use -Force only to replace this file."
        exit 0
    }
}

Write-Host "Downloading YOLOX-s ONNX model from: $ModelUrl"
Invoke-WebRequest -Uri $ModelUrl -OutFile $TargetFile
$Hash = (Get-FileHash -Path $TargetFile -Algorithm SHA256).Hash.ToLower()
$Size = (Get-Item -Path $TargetFile).Length
if ($Size -lt 1000000) { throw "Downloaded model is unexpectedly small ($Size bytes); remove it and try again." }
Write-Host "Installed: $TargetFile ($Size bytes)"
Write-Host "SHA-256  : $Hash"
