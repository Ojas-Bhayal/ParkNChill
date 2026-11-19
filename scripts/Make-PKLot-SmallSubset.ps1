# Create a small, annotation-preserving PKLot subset for ParkNChill.
# The source root must directly contain PUCPR, UFPR04 and UFPR05.
# Copies up to 8 image/XML pairs from the first 2 date folders in each lot
# (up to 6 independent lot/date groups and 48 full-scene images).

$ErrorActionPreference = 'Stop'

$sourceInput = Read-Host 'Enter extracted PKLot folder path that directly contains PUCPR, UFPR04 and UFPR05'
$sourceInput = $sourceInput.Trim().Trim('"')
if (-not (Test-Path -LiteralPath $sourceInput -PathType Container)) {
    throw "Source folder not found: $sourceInput"
}
$sourceRoot = (Resolve-Path -LiteralPath $sourceInput).Path.TrimEnd('\')
$destRoot = Join-Path (Get-Location).Path 'PKLot_Small'
$maxImagesPerGroup = 8
$sites = @('PUCPR', 'UFPR04', 'UFPR05')

if (Test-Path -LiteralPath $destRoot) {
    $answer = Read-Host "'$destRoot' already exists. Delete and recreate it? (Y/N)"
    if ($answer -notmatch '^(Y|y)$') { throw 'Cancelled; existing subset was not changed.' }
    Remove-Item -LiteralPath $destRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $destRoot | Out-Null

$totalImages = 0
$totalGroups = 0

foreach ($site in $sites) {
    $sitePath = Join-Path $sourceRoot $site
    if (-not (Test-Path -LiteralPath $sitePath -PathType Container)) {
        Write-Warning "Missing lot folder '$sitePath'; skipping this lot."
        continue
    }

    $dateFolders = Get-ChildItem -LiteralPath $sitePath -Directory -Recurse |
        Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' } |
        Sort-Object FullName |
        Select-Object -First 2

    foreach ($dateFolder in $dateFolders) {
        $pairs = @(
            Get-ChildItem -LiteralPath $dateFolder.FullName -File |
                Where-Object {
                    $_.Extension -match '^\.(jpg|jpeg|png)$' -and
                    (Test-Path -LiteralPath (Join-Path $dateFolder.FullName ($_.BaseName + '.xml')))
                } |
                Sort-Object Name
        )

        if ($pairs.Count -eq 0) {
            Write-Warning "No image/XML pairs found in '$($dateFolder.FullName)'; skipping."
            continue
        }

        # Select frames spread across the date rather than only adjacent captures.
        if ($pairs.Count -le $maxImagesPerGroup) {
            $selected = $pairs
        } else {
            $indices = for ($i = 0; $i -lt $maxImagesPerGroup; $i++) {
                [int][Math]::Round($i * ($pairs.Count - 1) / ($maxImagesPerGroup - 1))
            }
            $selected = @($indices | Select-Object -Unique | ForEach-Object { $pairs[$_] })
        }

        $relativeDir = $dateFolder.FullName.Substring($sourceRoot.Length).TrimStart('\')
        $destinationDir = Join-Path $destRoot $relativeDir
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null

        foreach ($imageFile in $selected) {
            Copy-Item -LiteralPath $imageFile.FullName -Destination (Join-Path $destinationDir $imageFile.Name)
            $xmlPath = Join-Path $dateFolder.FullName ($imageFile.BaseName + '.xml')
            Copy-Item -LiteralPath $xmlPath -Destination (Join-Path $destinationDir ([IO.Path]::GetFileName($xmlPath)))
            $totalImages++
        }
        $totalGroups++
        Write-Host ("Copied {0} image/XML pairs from {1}" -f $selected.Count, $relativeDir) -ForegroundColor Green
    }
}

Write-Host ''
Write-Host "Subset folder: $destRoot" -ForegroundColor Cyan
Write-Host "Image/XML pairs copied: $totalImages"
Write-Host "Date folders copied: $totalGroups"
if ($totalGroups -lt 3) {
    Write-Warning 'Fewer than 3 groups were found. Check that the selected source root contains PUCPR, UFPR04, and UFPR05 with date folders.'
} else {
    Write-Host 'Next: in ParkNChill, choose Import PKLot and select this PKLot_Small folder.' -ForegroundColor Cyan
}
Write-Host 'This subset is for workflow development. It is not enough by itself to claim general real-world accuracy.'
