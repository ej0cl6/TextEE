export RAMS_PATH="./RAMS_1.0c/"
export PROCESSED_DATA_PATH="../../processed_data/rams"

python process_rams.py -i $RAMS_PATH -o $PROCESSED_DATA_PATH --split split1
python process_rams.py -i $RAMS_PATH -o $PROCESSED_DATA_PATH --split split2
python process_rams.py -i $RAMS_PATH -o $PROCESSED_DATA_PATH --split split3
python process_rams.py -i $RAMS_PATH -o $PROCESSED_DATA_PATH --split split4
python process_rams.py -i $RAMS_PATH -o $PROCESSED_DATA_PATH --split split5