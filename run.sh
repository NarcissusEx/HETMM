dataset=MVTec_AD
tt=PTS
ts=60

# Generating template
# Original template
python run.py --mode temp --ttype ALL --dataset $dataset
# PTS template
python run.py --mode temp --ttype $tt --tsize $ts --dataset $dataset

# Anomaly detection and localization
# Original template
python run.py --mode test --ttype ALL --dataset $dataset
# PTS template
python run.py --mode test --ttype $tt --tsize $ts --dataset $dataset