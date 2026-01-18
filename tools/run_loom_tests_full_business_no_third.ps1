param(
    [string]$RunCmd = ""
)

$ErrorActionPreference = "Stop"

# Жёстко строим абсолютный путь к run.cmd как "../run.cmd" относительно каталога скрипта
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultRun = Join-Path $scriptDir "..\run.cmd"
if ([string]::IsNullOrWhiteSpace($RunCmd)) {
    $RunCmd = $defaultRun
}
elseif (-not (Test-Path $RunCmd) -and (Test-Path $defaultRun)) {
    $RunCmd = $defaultRun
}

$tests = @(
    @{ Name = "full_business_no_third";
       TEST_INPUT_FILE = "example/test_in.json";
       APPLY_ZERO_PER_DAY_LIMIT = "true";
       APPLY_ZERO_PER_MACHINE_LIMIT = "false";
       APPLY_INDEX_UP = "false";
       APPLY_PROP_OBJECTIVE = "true";
       APPLY_STRATEGY_PENALTY = "true";
       APPLY_THIRD_ZERO_BAN = "false";
       APPLY_QTY_MINUS = "false";
       USE_GREEDY_HINT = "true";
    }
)

# Общие настройки: обычный режим без enumerate и увеличенный лимит времени
$env:CALC_TEST_DATA       = "true"
$env:SOLVER_ENUMERATE     = "false"
$env:SOLVER_ENUMERATE_COUNT = "1"
$env:APPLY_DOWNTIME_LIMITS = "true"
$env:LOOM_MAX_TIME        = "1800"

if (!(Test-Path "log")) {
    New-Item -ItemType Directory -Path "log" | Out-Null
}

foreach ($t in $tests) {
    Write-Host "=== Running $($t.Name) ===" -ForegroundColor Cyan

    # Устанавливаем специфические переменные окружения
    $env:TEST_INPUT_FILE              = $t.TEST_INPUT_FILE
    $env:APPLY_ZERO_PER_DAY_LIMIT     = $t.APPLY_ZERO_PER_DAY_LIMIT
    $env:APPLY_ZERO_PER_MACHINE_LIMIT = $t.APPLY_ZERO_PER_MACHINE_LIMIT
    $env:APPLY_INDEX_UP               = $t.APPLY_INDEX_UP
    $env:APPLY_PROP_OBJECTIVE         = $t.APPLY_PROP_OBJECTIVE
    $env:APPLY_STRATEGY_PENALTY       = $t.APPLY_STRATEGY_PENALTY
    $env:APPLY_THIRD_ZERO_BAN         = $t.APPLY_THIRD_ZERO_BAN
    $env:APPLY_QTY_MINUS              = $t.APPLY_QTY_MINUS
    $env:USE_GREEDY_HINT              = $t.USE_GREEDY_HINT

    # Чистим основной лог
    if (Test-Path "log/aps-loom.log") {
        Remove-Item "log/aps-loom.log"
    }

    # Запуск расчёта
    & $RunCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Test $($t.Name) failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }

    # Копируем лог под уникальным именем
    $targetLog = "log/aps-loom_$($t.Name).log"
    if (Test-Path "log/aps-loom.log") {
        Copy-Item "log/aps-loom.log" $targetLog -Force
        Write-Host "Saved log to $targetLog" -ForegroundColor Green
    } else {
        Write-Host "Warning: log/aps-loom.log not found after $($t.Name)" -ForegroundColor Yellow
    }
}
