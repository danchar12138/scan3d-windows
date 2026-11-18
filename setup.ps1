# Downloads the pinned components into .\tools. Safe to re-run: finished components are skipped,
# interrupted downloads resume.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$tools = Join-Path $root 'tools'
$cache = Join-Path $tools '_downloads'

# Use the copies that ship with Windows. A bare "tar.exe" can resolve to Git's GNU tar when this
# script is launched from Git Bash, and GNU tar reads "C:\..." as a remote host.
$curl = Join-Path $env:SystemRoot 'System32\curl.exe'
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'

function Get-File($url, $dest) {
    # GitHub's and Hugging Face's CDNs drop long transfers on some networks; retry on any error
    # and resume each time.
    & $curl -L --fail --retry 30 --retry-delay 3 --retry-all-errors -C - -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
}

# Programs, each a zip that unpacks into its own folder.
$programs = @(
    # Runs the code in .\app; the embeddable build needs no installation.
    @{ name = 'python'
       url = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' },
    # Spirula Studio: frames, masks, camera poses, priors, splat training, surface extraction.
    @{ name = 'spirula'
       url = 'https://github.com/harry7557558/spirula-studio/releases/download/v2026.11.12/spirula-2026.11.12-windows-vulkan-x86_64.zip' },
    # COLMAP (CUDA build): undistorted image workspace and mesh texturing from the photos.
    @{ name = 'colmap'
       url = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip' }
)

New-Item -ItemType Directory -Force -Path $cache | Out-Null

foreach ($c in $programs) {
    $dest = Join-Path $tools $c.name
    $marker = Join-Path $dest '.installed'
    if (Test-Path $marker) { Write-Host "[skip] $($c.name) already installed"; continue }

    $zip = Join-Path $cache ([IO.Path]::GetFileName($c.url))
    Write-Host "[download] $($c.name)"
    Get-File $c.url $zip

    Write-Host "[extract] $($c.name)"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    & $tar -xf $zip -C $dest
    if ($LASTEXITCODE -ne 0) { throw "extract failed: $zip" }

    # Some archives wrap everything in a single top-level folder; flatten it.
    $entries = @(Get-ChildItem -Force $dest)
    if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) {
        $inner = $entries[0].FullName
        Get-ChildItem -Force $inner | Move-Item -Destination $dest
        Remove-Item -Force $inner
    }

    Set-Content -Path $marker -Value $c.url
    Remove-Item -Force $zip
    Write-Host "[ok] $($c.name)"
}

# Network weights (tools\models). app\models.py knows which files each stage needs and where
# they come from. This fetches the ones the default template uses; any other weights download
# by themselves when a stage that needs them starts (the licence that covers them is recorded
# then).
Write-Host '[models] checking network weights'
& (Join-Path $tools 'python\python.exe') (Join-Path $root 'app\scan3d.py') models --download
if ($LASTEXITCODE -ne 0) { throw 'model download failed' }

Write-Host 'All components ready.'
