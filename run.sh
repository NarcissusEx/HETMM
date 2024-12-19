dataset=MVTec_AD
tt=PTS
ts=60

# Generating template
# Original template
python run.py --mode temp --ttype ALL --dataset $dataset --datapath <data_path>
# PTS template
python run.py --mode temp --ttype $tt --tsize $ts --dataset $dataset --datapath <data_path>

# Anomaly detection and localization
# Original template
python run.py --mode test --ttype ALL --dataset $dataset --datapath <data_path>
# PTS template
python run.py --mode test --ttype $tt --tsize $ts --dataset $dataset --datapath <data_path>