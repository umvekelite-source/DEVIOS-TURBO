param(
    [string]$Endpoint = "http://127.0.0.1:18080",
    [int]$Tokens = 128
)

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " DEVIOS Turbo -- Latency & Speculation Benchmark" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Check models
try {
    $models = Invoke-RestMethod -Uri "$Endpoint/v1/models" -Method Get -TimeoutSec 5
    Write-Host "[+] Connected to DEVIOS Turbo" -ForegroundColor Green
    Write-Host "[+] Active Model: $($models.data[0].id)" -ForegroundColor White
} catch {
    Write-Host "[-] Failed to connect to $Endpoint. Is the server running?" -ForegroundColor Red
    exit 1
}

$prompt = @{
    model = $models.data[0].id
    messages = @(
        @{
            role = "system"
            content = "You are OpenCode assistant.`n`n### Summary`nTask completed.`n`n### Work State`n- Status: Completed`n- Phase: Done`n`n### Decisions`n1. Zero latency threadpool.`n2. Q4_0 KV-cache.`n"
        },
        @{
            role = "user"
            content = "Generate the next work state report with Summary, Work State, Decisions:"
        }
    )
    max_tokens = $Tokens
    temperature = 0.0
    stream = $false
} | ConvertTo-Json -Depth 5

Write-Host "`n[*] Running generation benchmark ($Tokens tokens)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$res = Invoke-RestMethod -Uri "$Endpoint/v1/chat/completions" -Method Post -Body $prompt -ContentType "application/json"
$sw.Stop()

$dt = $res.devios_turbo
Write-Host "`n================ BENCHMARK RESULTS ================" -ForegroundColor Green
Write-Host ("Generation Speed : {0:N2} tok/s" -f $dt.tokens_per_sec) -ForegroundColor Cyan
Write-Host ("Acceptance Rate  : {0:P1}" -f $dt.acceptance_rate) -ForegroundColor Cyan
Write-Host ("Avg Accepted/Step: {0:N2} tokens per weight read" -f $dt.avg_accepted_per_step) -ForegroundColor Cyan
Write-Host ("Peak Draft Window: K = {0}" -f $dt.final_k) -ForegroundColor Cyan
Write-Host ("Total Wall Time  : {0:N2} s" -f ($sw.ElapsedMilliseconds / 1000.0)) -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Green
