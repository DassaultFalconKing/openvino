param(
  [Parameter(Mandatory)] [string]$Runtime,
  [Parameter(Mandatory)] [string]$Tag,
  [int]$MaxTokens = 64
)
$ErrorActionPreference = "Stop"
$ev = "C:\gm\orchestration\decode-regression-20261022"
$model = "C:\Users\testc\qwen36-35b-a3b-baseline\models\Qwen3.6-35B-A3B-int4-ov"
$py = "C:\gm\otd-research\build-tools\python312\python.exe"
$port = 9199
Get-Process ovms -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$run = "$ev\runs\$Tag-$stamp"
$cacheKey = ((Get-FileHash "$Runtime\openvino_intel_gpu_plugin.dll").Hash.Substring(0, 12))
$cache = "$ev\cache\$cacheKey"
New-Item -ItemType Directory -Force $run, $cache | Out-Null
$wp = ($model.Replace('\', '/') + '/openvino_language_model.bin')
$plugin = "{`"OFFLOAD_RATIO`":0,`"WEIGHTS_PATH`":`"$wp`",`"CACHE_DIR`":`"$($cache.Replace('\','/'))`"}"
$argList = @(
  "--model_path", $model, "--model_name", "qwen",
  "--target_device", "GPU", "--task", "text_generation",
  "--max_num_batched_tokens", "16384", "--cache_interval_multiplier", "64",
  "--plugin_config", ('"' + $plugin.Replace('"', '\"') + '"'),
  "--log_level", "INFO", "--log_path", "$run\ovms.log",
  "--rest_port", "$port", "--rest_bind_address", "127.0.0.1"
)
$savedPath = $env:PATH
$env:PATH = "$Runtime;$Runtime\python;$env:PATH"
$proc = Start-Process -FilePath "$Runtime\ovms.exe" -ArgumentList $argList -WorkingDirectory $Runtime -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput "$run\stdout.log" -RedirectStandardError "$run\stderr.log"
$env:PATH = $savedPath

$ready = $false
for ($i=0; $i -lt 180; $i++) {
  try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v2/health/ready" -TimeoutSec 1
    $ready = $true; break
  } catch { Start-Sleep 1 }
}
if (-not $ready) { Stop-Process -Id $proc.Id -Force -EA SilentlyContinue; throw "not ready" }

# warmup
& $py "C:\gm\otd-research\build-tools\bench_otd.py" --port $port --model qwen --max-tokens 16 --repeats 0 --out "$run\warmup.json" | Out-Null

$samples = New-Object System.Collections.Generic.List[object]
$job = Start-Job -ScriptBlock {
  param($out)
  $rows = @()
  for ($i=0; $i -lt 40; $i++) {
    $p = Get-Process ovms -EA SilentlyContinue
    $gpu = 0
    try {
      $c = Get-Counter '\GPU Engine(*)\Utilization Percentage' -EA SilentlyContinue
      if ($c) { $gpu = ($c.CounterSamples | Measure-Object CookedValue -Maximum).Maximum }
    } catch {}
    $rows += [pscustomobject]@{
      t = [DateTime]::UtcNow.ToString("o")
      gpu_max = [math]::Round($gpu,1)
      cpu = if($p){[math]::Round($p.CPU,1)}else{-1}
      ws_mb = if($p){[math]::Round($p.WorkingSet64/1MB)}else{-1}
    }
    Start-Sleep -Milliseconds 400
  }
  $rows | ConvertTo-Json | Set-Content $out
} -ArgumentList "$run\util.json"

Start-Sleep 1
& $py "C:\gm\otd-research\build-tools\bench_otd.py" --port $port --model qwen --max-tokens $MaxTokens --repeats 1 --out "$run\bench.json" | Out-Null
Wait-Job $job -Timeout 30 | Out-Null
Stop-Job $job -EA SilentlyContinue
Receive-Job $job -EA SilentlyContinue | Out-Null

$j = Get-Content "$run\bench.json" -Raw | ConvertFrom-Json
$rate = [math]::Round($j.runs[0].decode_tok_per_s_after_first, 2)
Stop-Process -Id $proc.Id -Force -EA SilentlyContinue
Start-Sleep 2
$util = @()
if (Test-Path "$run\util.json") { $util = Get-Content "$run\util.json" -Raw | ConvertFrom-Json }
$gpuVals = @($util | ForEach-Object { $_.gpu_max })
$wsVals = @($util | ForEach-Object { $_.ws_mb })
"RESULT tag=$Tag rate=$rate gpu_max=$($gpuVals | Measure-Object -Maximum | % Maximum) gpu_avg=$([math]::Round(($gpuVals | Measure-Object -Average).Average,1)) ws_mb=$($wsVals | Measure-Object -Maximum | % Maximum) n=$($gpuVals.Count) run=$run"
