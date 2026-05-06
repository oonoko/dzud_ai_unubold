#!/usr/bin/env pwsh
# Модель статистик харуулах

Write-Host "🚀 Зуд AI - Модель Статистик" -ForegroundColor Cyan
Write-Host ""

# Python байгаа эсэхийг шалгах
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python олдсонгүй. Python суулгана уу." -ForegroundColor Red
    exit 1
}

# Шаардлагатай файлууд байгаа эсэхийг шалгах
$requiredFiles = @(
    "dzud_ai_dataset_advanced.csv",
    "dzud_risk_model_advanced.pkl",
    "scaler_advanced.pkl",
    "model_metadata_advanced.json"
)

$missing = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host "❌ Дараах файлууд олдсонгүй:" -ForegroundColor Red
    foreach ($file in $missing) {
        Write-Host "   - $file" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "💡 Эхлээд модель сургана уу:" -ForegroundColor Cyan
    Write-Host "   python train_model_advanced.py" -ForegroundColor White
    exit 1
}

# Статистик харуулах
Write-Host "📊 Статистик уншиж байна..." -ForegroundColor Green
Write-Host ""

python show_model_stats.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Алдаа гарлаа" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Амжилттай" -ForegroundColor Green
