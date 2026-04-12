param(
    [string]$OutputDir = "dist\clean-package",
    [string]$ArchiveBaseName = "hf-gaia-agent-clean"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ResolvedOutputDir = Join-Path $RepoRoot $OutputDir
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ArchivePath = Join-Path $ResolvedOutputDir "$ArchiveBaseName-$Timestamp.zip"
$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("hf-gaia-agent-clean-" + [guid]::NewGuid().ToString("N"))

$ExcludedNames = @(
    ".git",
    ".env",
    ".venv",
    ".uv-cache",
    ".cache",
    ".pytest_cache",
    ".runtime-artifacts",
    ".test-artifacts",
    ".tmp",
    ".tmp_pytest",
    "__pycache__",
    "pytest-cache-files-lkr7tsdo",
    "pytest-cache-files-v71sdjtx",
    "testtmpbase",
    "dist"
)

function Copy-CleanTree {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    $Item = Get-Item -LiteralPath $SourcePath -Force
    if ($ExcludedNames -contains $Item.Name) {
        return
    }

    if ($Item.PSIsContainer) {
        New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
        Get-ChildItem -LiteralPath $Item.FullName -Force | ForEach-Object {
            Copy-CleanTree -SourcePath $_.FullName -DestinationPath (Join-Path $DestinationPath $_.Name)
        }
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestinationPath) | Out-Null
    Copy-Item -LiteralPath $Item.FullName -Destination $DestinationPath -Force
}

try {
    New-Item -ItemType Directory -Force -Path $ResolvedOutputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

    Get-ChildItem -LiteralPath $RepoRoot -Force | ForEach-Object {
        Copy-CleanTree -SourcePath $_.FullName -DestinationPath (Join-Path $StageDir $_.Name)
    }

    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }

    Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ArchivePath
    Write-Output $ArchivePath
}
finally {
    if (Test-Path -LiteralPath $StageDir) {
        Remove-Item -LiteralPath $StageDir -Recurse -Force
    }
}
