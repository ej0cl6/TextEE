export GENEVA_PATH="./all_data.json"
export PROCESSED_DATA_PATH="../../processed_data/geneva"

python process_geneva.py -i $GENEVA_PATH -o $PROCESSED_DATA_PATH --split split1
python process_geneva.py -i $GENEVA_PATH -o $PROCESSED_DATA_PATH --split split2
python process_geneva.py -i $GENEVA_PATH -o $PROCESSED_DATA_PATH --split split3
python process_geneva.py -i $GENEVA_PATH -o $PROCESSED_DATA_PATH --split split4
python process_geneva.py -i $GENEVA_PATH -o $PROCESSED_DATA_PATH --split split5
