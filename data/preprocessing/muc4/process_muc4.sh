export MUC4_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/muc4"

python process_muc4.py -i $MUC4_PATH -o $PROCESSED_DATA_PATH --split split1
python process_muc4.py -i $MUC4_PATH -o $PROCESSED_DATA_PATH --split split2
python process_muc4.py -i $MUC4_PATH -o $PROCESSED_DATA_PATH --split split3
python process_muc4.py -i $MUC4_PATH -o $PROCESSED_DATA_PATH --split split4
python process_muc4.py -i $MUC4_PATH -o $PROCESSED_DATA_PATH --split split5

