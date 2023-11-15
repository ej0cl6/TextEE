export CASIE_PATH="./CASIE/data/annotation/"
export PROCESSED_DATA_PATH="../../processed_data/casie"

python process_casie.py -i $CASIE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split1
python process_casie.py -i $CASIE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split2
python process_casie.py -i $CASIE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split3
python process_casie.py -i $CASIE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split4
python process_casie.py -i $CASIE_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --seg_map seg_map.json --split split5
