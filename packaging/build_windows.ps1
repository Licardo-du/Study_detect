param(
    [switch]$SkipInstall,
    [string]$Python = "python"
)

# Stop immediately when any build step fails.
$ErrorActionPreference = "Stop"

# Resolve the project root so the script can be launched from any directory.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $Python"

# Install runtime and build dependencies unless the caller already prepared the env.
if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
    & $Python -m pip install -r requirements-build.txt
}

# Compile the main modules before packaging so syntax errors fail early.
& $Python -m py_compile app.py ui.py ai_core.py auth.py db.py network_utils.py visualization.py path_utils.py

# Build the directory-style Windows distribution from the spec file.
& $Python -m PyInstaller --clean --noconfirm packaging\StudyBehaviorMonitor.spec

Write-Host ""
Write-Host "Build finished."
Write-Host "Executable: $ProjectRoot\dist\StudyBehaviorMonitor\StudyBehaviorMonitor.exe"
