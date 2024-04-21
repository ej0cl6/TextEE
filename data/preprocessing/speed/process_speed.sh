export SPEED_PATH="./covid-only_speed.json"
export PROCESSED_DATA_PATH="../../processed_data/speed"

python process_speed.py -i $SPEED_PATH -o $PROCESSED_DATA_PATH --split split1
python process_speed.py -i $SPEED_PATH -o $PROCESSED_DATA_PATH --split split2
python process_speed.py -i $SPEED_PATH -o $PROCESSED_DATA_PATH --split split3
python process_speed.py -i $SPEED_PATH -o $PROCESSED_DATA_PATH --split split4
python process_speed.py -i $SPEED_PATH -o $PROCESSED_DATA_PATH --split split5
