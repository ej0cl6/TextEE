export FEWEVENT_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/fewevent"

python process_fewevent.py -i $FEWEVENT_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --split split1
python process_fewevent.py -i $FEWEVENT_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --split split2
python process_fewevent.py -i $FEWEVENT_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --split split3
python process_fewevent.py -i $FEWEVENT_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --split split4
python process_fewevent.py -i $FEWEVENT_PATH -o $PROCESSED_DATA_PATH --token_map token_map.json --split split5
