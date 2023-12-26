export GENIA2011_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/genia2011"

python process_genia.py -i $GENIA2011_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split1
python process_genia.py -i $GENIA2011_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split2
python process_genia.py -i $GENIA2011_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split3
python process_genia.py -i $GENIA2011_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split4
python process_genia.py -i $GENIA2011_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split5

