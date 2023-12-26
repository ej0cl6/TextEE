export MLEE_PATH="./MLEE-1.0.2-rev1/standoff/full/"
export PROCESSED_DATA_PATH="../../processed_data/mlee"

python process_mlee.py -i $MLEE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split1
python process_mlee.py -i $MLEE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split2
python process_mlee.py -i $MLEE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split3
python process_mlee.py -i $MLEE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split4
python process_mlee.py -i $MLEE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split5
