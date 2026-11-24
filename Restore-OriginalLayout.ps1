[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $Destination
)

$ErrorActionPreference = 'Stop'
$packageRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$destinationRoot = [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')
$originalRoot = 'D:\MMOE_UWL_cf'

function Test-PathInside([string] $Child, [string] $Parent) {
    return $Child.StartsWith($Parent + '\', [System.StringComparison]::OrdinalIgnoreCase)
}

if ((Test-Path -LiteralPath $destinationRoot) -or
    (Test-PathInside $destinationRoot $packageRoot) -or
    (Test-PathInside $destinationRoot $originalRoot)) {
    throw 'Choose a new directory outside this package and outside the original project.'
}

$rows = Import-Csv -LiteralPath (Join-Path $packageRoot 'FILE_INVENTORY.csv') |
    Where-Object { $_.restore_original_layout -eq 'True' }
$seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$plan = foreach ($row in $rows) {
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $row.package_path))
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $destinationRoot $row.original_path))
    if (-not (Test-PathInside $sourcePath $packageRoot) -or
        -not (Test-PathInside $outputPath $destinationRoot) -or
        -not $seen.Add($outputPath)) {
        throw "Invalid or duplicate restore path for $($row.package_path)"
    }
    if ((Get-FileHash -LiteralPath $sourcePath).Hash -ne $row.sha256) {
        throw "Checksum mismatch for $($row.package_path)"
    }
    [pscustomobject]@{ Source = $sourcePath; Destination = $outputPath }
}

if (-not $plan) { throw 'No files selected for restoration.' }
$null = New-Item -ItemType Directory -Path $destinationRoot
foreach ($file in $plan) {
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $file.Destination) -Force
    Copy-Item -LiteralPath $file.Source -Destination $file.Destination
}
Write-Output "Restored $($plan.Count) files to $destinationRoot."
