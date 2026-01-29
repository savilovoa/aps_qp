param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

# Каталог скрипта (для относительных путей)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir\..

if (!(Test-Path "log")) {
    New-Item -ItemType Directory -Path "log" | Out-Null
}

# Общие настройки для всех запусков: LONG_SIMPLE + LS_PROP + noIndex + H123
$env:CALC_TEST_DATA            = "true"
$env:SOLVER_ENUMERATE          = "false"
$env:SOLVER_ENUMERATE_COUNT    = "1"
$env:HORIZON_MODE              = "LONG_SIMPLE"
$env:APPLY_QTY_MINUS           = "true"
$env:APPLY_PROP_OBJECTIVE      = "true"
$env:APPLY_OVERPENALTY_INSTEAD_OF_PROP = "false"
$env:SIMPLE_USE_PROP_MULT      = "true"
$env:APPLY_STRATEGY_PENALTY    = "true"
$env:APPLY_INDEX_UP            = "true"
$env:SIMPLE_DISABLE_INDEX_UP   = "true"
$env:SIMPLE_DEBUG_H_START      = "true"
$env:SIMPLE_DEBUG_H_MODE       = "H123"
# Входные данные по умолчанию
$env:TEST_INPUT_FILE           = "example/test_in.json"

$timeConfigs = @(
    @{ Name = "t600";  LOOM_MAX_TIME = "600"  },
    @{ Name = "t1800"; LOOM_MAX_TIME = "1800" }
)

foreach ($cfg in $timeConfigs) {
    $name = $cfg.Name
    $tsec = $cfg.LOOM_MAX_TIME
    Write-Host "=== Running LS_PROP_noIndex_H123 with LOOM_MAX_TIME=$tsec ($name) ===" -ForegroundColor Cyan

    $env:LOOM_MAX_TIME = $tsec

    # Удаляем предыдущий лог анализа, если есть
    $analysisLog = "log/analyze_ls_prop_noIndex_H123.log"
    if (Test-Path $analysisLog) {
        Remove-Item $analysisLog -Force
    }

    # Запуск анализа
    & $PythonExe -m tools.analyze_ls_prop_noIndex_H123_by_product | Tee-Object -FilePath $analysisLog
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Run $name failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }

    # Копируем лог под уникальным именем
    $targetLog = "log/analyze_ls_prop_noIndex_H123_$name.log"
    if (Test-Path $analysisLog) {
        Copy-Item $analysisLog $targetLog -Force
        Write-Host "Saved analysis log to $targetLog" -ForegroundColor Green
    } else {
        Write-Host "Warning: $analysisLog not found after $name" -ForegroundColor Yellow
    }
}
