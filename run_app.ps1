$ErrorActionPreference = "Stop"

$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $bundled = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $bundled) {
        $python = $bundled
    } else {
        throw "Python was not found. Install Python 3.11+ and rerun this script."
    }
}

$hasStreamlit = & $python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('streamlit') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit is not installed for this Python. Run: pip install -r requirements.txt"
}

& $python -m streamlit run app.py
