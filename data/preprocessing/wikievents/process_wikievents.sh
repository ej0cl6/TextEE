export WIKIEVENTS_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/wikievents"

python process_wikievents.py -i $WIKIEVENTS_PATH -o $PROCESSED_DATA_PATH --seg_map seg_map.json --split split1
python process_wikievents.py -i $WIKIEVENTS_PATH -o $PROCESSED_DATA_PATH --seg_map seg_map.json --split split2
python process_wikievents.py -i $WIKIEVENTS_PATH -o $PROCESSED_DATA_PATH --seg_map seg_map.json --split split3
python process_wikievents.py -i $WIKIEVENTS_PATH -o $PROCESSED_DATA_PATH --seg_map seg_map.json --split split4
python process_wikievents.py -i $WIKIEVENTS_PATH -o $PROCESSED_DATA_PATH --seg_map seg_map.json --split split5

