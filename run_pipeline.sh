#!/bin/bash

set -e

echo "========================================"
echo "1. Auditing raw ESOL data"
echo "========================================"
python src/01_audit_data.py

echo
echo "========================================"
echo "2. Curating molecular dataset"
echo "========================================"
python src/02_curate_data.py

echo
echo "========================================"
echo "3. Generating RDKit molecular features"
echo "========================================"
python src/03_featurize.py

echo
echo "========================================"
echo "4. Chemical cluster split"
echo "========================================"
python src/04b_cluster_split.py

echo
echo "========================================"
echo "5. Representation comparison"
echo "========================================"
python src/05_train_rf_comparison.py

echo
echo "========================================"
echo "6. Hyperparameter tuning"
echo "========================================"
python src/06_tune_rf.py

echo
echo "========================================"
echo "7. Validation error analysis"
echo "========================================"
python src/07_validation_error_analysis.py

echo
echo "========================================"
echo "8. Feature interpretation"
echo "========================================"
python src/08_descriptor_importance.py

echo
echo "========================================"
echo "9. Final test evaluation"
echo "========================================"
python src/09_final_test.py

echo
echo "========================================"
echo "10. Final analysis"
echo "========================================"
python src/10_final_analysis.py

echo
echo "Pipeline complete."