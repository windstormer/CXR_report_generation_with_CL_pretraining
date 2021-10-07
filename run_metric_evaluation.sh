source /home/nlp/miniconda3/etc/profile.d/conda.sh

cd metrics
rm *.json
rm *.log
cd ..

conda activate torch
python evaluate_all.py

cd metrics
cd CaptionMetrics
conda activate py27
python run_metric_evaluation.py

conda activate torch
python visualize_metrics.py
