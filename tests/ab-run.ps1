param(
  [Parameter(Mandatory)] [string]$Runtime,
  [Parameter(Mandatory)] [string]$Tag,
  [string[]]$ExtraArgs = @(),
  [hashtable]$Env = @{},
  [int]$Repeats = 3,
  [int]$MaxTokens = 256,
  [int]$OffloadRatio = 20,
  [string]$PluginExtra = ""
)
$ErrorActionPreference = "Stop"
$ev = "C:\gm\orchestration\decode-regression-20261022"
$model = "C:\Users\testc\qwen36-35b-a3b-baseline\models\Qwen3.6-35B-A3B-int4-ov"
$py = "C:\gm\otd-research\build-tools\python312\python.exe"
$port = 9188

Get-Process ovms -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 2
$freeMiB = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
if ($freeMiB -lt 11000) { throw "RAM $freeMiB" }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$run = "$ev\runs\$Tag-$stamp"
$cacheKey = ((Get-FileHash "$Runtime\openvino_intel_gpu_plugin.dll").Hash.Substring(0, 12))
$cache = "$ev\cache\$cacheKey"
New-Item -ItemType Directory -Force $run, $cache | Out-Null

foreach ($v in 'MOE_OTD_TRACE', 'MOE_OTD_PERF_LOG', 'GPU_MOE_OTD_PACK_DIR', 'GPU_MOE_OTD_PACK_MODE', 'OTD_RESIDENT_USM_HOST', 'PYTHONPATH', 'OV_GPU_VERBOSE') {
  Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
if (Test-Path "$Runtime\python") { $env:PYTHONHOME = "$Runtime\python" } else { Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue }
foreach ($k in $Env.Keys) { Set-Item "Env:$k" $Env[$k] }

$wp = ($model.Replace('\', '/') + '/openvino_language_model.bin')
$plugin = "{`"OFFLOAD_RATIO`":$OffloadRatio,`"WEIGHTS_PATH`":`"$wp`",`"CACHE_DIR`":`"$($cache.Replace('\','/'))`"$PluginExtra}"
$argList = @(
  "--model_path", $model, "--model_name", "qwen",
  "--target_device", "GPU", "--task", "text_generation",
  "--tool_parser", "qwen3coder", "--reasoning_parser", "qwen3",
  "--max_num_batched_tokens", "16384", "--cache_interval_multiplier", "64",
  "--plugin_config", ('"' + $plugin.Replace('"', '\"') + '"'),
  "--log_level", "INFO", "--log_path", "$run\ovms.log",
  "--rest_port", "$port", "--rest_bind_address", "127.0.0.1"
) + $ExtraArgs

$savedPath = $env:PATH
$env:PATH = "$Runtime;$Runtime\python;$env:PATH"
$proc = Start-Process -FilePath "$Runtime\ovms.exe" -ArgumentList $argList -WorkingDirectory $Runtime -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput "$run\stdout.log" -RedirectStandardError "$run\stderr.log"
$env:PATH = $savedPath

[ordered]@{
  tag = $Tag; runtime = $Runtime; pid = $proc.Id; free_mib = $freeMiB; extra_args = $ExtraArgs; env = $Env
  plugin_config = $plugin
  ovms_sha = (Get-FileHash "$Runtime\ovms.exe").Hash
  gpu_plugin_sha = (Get-FileHash "$Runtime\openvino_intel_gpu_plugin.dll").Hash
  openvino_sha = (Get-FileHash "$Runtime\openvino.dll").Hash
  started_at = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json | Set-Content "$run\run.json"

& $py "C:\gm\otd-research\build-tools\bench_otd.py" --port $port --model qwen --max-tokens $MaxTokens --repeats $Repeats --out "$run\bench.json" | Out-Null

$j = Get-Content "$run\bench.json" -Raw | ConvertFrom-Json
$rates = @($j.runs | ForEach-Object { [math]::Round($_.decode_tok_per_s_after_first, 2) })
$alive = -not $proc.HasExited
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 2
"RESULT tag=$Tag ready=$($j.ready) ready_wait_s=$([math]::Round($j.ready_wait_s,1)) warm_ttft=$([math]::Round($j.warmup.ttft_s,2)) runs=$($rates -join ',') alive=$alive run=$run"
