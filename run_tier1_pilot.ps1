<#
  Tier 1 pilot (~300 papers) — clean-slate, relevance-gated build.
  Run this from YOUR OWN terminal so your API key stays in your environment.

  1. Set your Semantic Scholar key first (kept in env, never written to disk):
       $env:SEMANTIC_SCHOLAR_API_KEY = "<your key>"
     (Or place it in research_rag/data/s2_api_key.txt — the code reads it. Your
      choice; keyless also works but snowball will be slow with 429s.)

  2. First run — wipe taxonomy + vector store, then build:
       .\run_tier1_pilot.ps1 -Reset
     Resume after an interruption (Ctrl+C / reboot) — do NOT pass -Reset:
       .\run_tier1_pilot.ps1
     (downloads dedup, classify/index skip done papers, snowball resumes from
      its checkpoint — so re-running continues instead of restarting.)

  Expect ~18-20h of CPU on this machine. Progress logs: data/logs/nightly.log
#>
param(
    [switch]$Reset,
    [int]$MaxPerQuery = 15,
    [int]$SnowballCap = 300
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
# Unbuffered so python output streams to the console / Tee-Object log line by
# line instead of sitting in an 8KB block buffer -- otherwise a long run looks
# "frozen" at the last header even though it is working.
$env:PYTHONUNBUFFERED = "1"
$py   = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$main = Join-Path $PSScriptRoot "main.py"

# S2 key status is reported authoritatively by the 'healthcheck' preflight
# below (it reads the SAME loader the pipeline uses: env var OR
# data/s2_api_key.txt). We do NOT hand-roll an env-only check here -- the old
# one falsely warned "not set" when the key was correctly placed in the file.

$queries = @(
    "quantum illumination Lloyd Gaussian",
    "microwave quantum illumination Josephson",
    "JTWPA traveling wave parametric amplifier",
    "two-mode squeezed vacuum microwave",
    "phase conjugate receiver quantum radar",
    "Rydberg atom RF receiver radar",
    "electromagnetically induced transparency Autler-Townes",
    "quantum LiDAR entangled photons",
    "quantum target detection Helstrom Chernoff",
    "quantum pulse compression ranging",
    "spontaneous parametric down conversion entanglement",
    "quantum hypothesis testing target discrimination",
    "microwave optical transducer quantum",
    "quantum radar range limitation",
    "circuit QED superconducting entanglement"
)

Write-Host "=== Preflight: Ollama health ===" -ForegroundColor Cyan
& $py $main healthcheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "ABORTING: Ollama preflight failed (see message above). Nothing was downloaded or wiped." -ForegroundColor Red
    exit 1
}

if ($Reset) {
    Write-Host "=== Clean slate (reset taxonomy + vector store) ===" -ForegroundColor Cyan
    & $py $main reset_taxonomy
    & $py $main reset_index
}

Write-Host "=== Gated downloads (no forced tier; off-topic filtered) ===" -ForegroundColor Cyan
foreach ($q in $queries) {
    Write-Host ">>> download: $q"
    & $py $main download $q --max-results $MaxPerQuery --gate
}

Write-Host "=== Extract (PDF text + metadata) ===" -ForegroundColor Cyan
foreach ($q in $queries) {
    Write-Host ">>> extract: $q"
    & $py $main extract $q
}

Write-Host "=== Classify + index (resumable) ===" -ForegroundColor Cyan
foreach ($q in $queries) {
    Write-Host ">>> classify/index: $q"
    & $py $main classify $q
    & $py $main index $q
}

Write-Host "=== Snowball top-up to $SnowballCap (nightly: checkpoint/throttle/notify) ===" -ForegroundColor Cyan
& $py $main snowball --tier 1 --max-papers $SnowballCap --nightly

Write-Host "=== Final stats ===" -ForegroundColor Cyan
& $py $main pipeline_stats
& $py $main taxonomy
