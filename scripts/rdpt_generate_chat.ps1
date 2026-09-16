# Launch the Rakshak Gradio demo from the repo root with the project venv.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:LLM_API_BASE) { $env:LLM_API_BASE = "http://localhost:11434/v1" }
if (-not $env:LLM_API_KEY) { $env:LLM_API_KEY = "ollama" }
if (-not $env:LLM_MODEL) { $env:LLM_MODEL = "mistral" }
python -m demo.app @args
