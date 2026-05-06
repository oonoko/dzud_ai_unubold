#!/bin/bash
# Модель статистик харуулах

echo "🚀 Зуд AI - Модель Статистик"
echo ""

# Python байгаа эсэхийг шалгах
if ! command -v python3 &> /dev/null; then
    echo "❌ Python олдсонгүй. Python суулгана уу."
    exit 1
fi

# Шаардлагатай файлууд байгаа эсэхийг шалгах
required_files=(
    "dzud_ai_dataset_advanced.csv"
    "dzud_risk_model_advanced.pkl"
    "scaler_advanced.pkl"
    "model_metadata_advanced.json"
)

missing=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing+=("$file")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "❌ Дараах файлууд олдсонгүй:"
    for file in "${missing[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "💡 Эхлээд модель сургана уу:"
    echo "   python3 train_model_advanced.py"
    exit 1
fi

# Статистик харуулах
echo "📊 Статистик уншиж байна..."
echo ""

python3 show_model_stats.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Алдаа гарлаа"
    exit 1
fi

echo ""
echo "✅ Амжилттай"
