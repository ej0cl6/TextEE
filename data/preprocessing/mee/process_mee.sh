export MEE_PATH="./MEE/"
export SPLIT_PATH="./split-en/"
export PROCESSED_DATA_PATH="../../processed_data/mee-en"

python process_mee.py -i $MEE_PATH -o $PROCESSED_DATA_PATH --lang english --split_path $SPLIT_PATH --split split1
python process_mee.py -i $MEE_PATH -o $PROCESSED_DATA_PATH --lang english --split_path $SPLIT_PATH --split split2
python process_mee.py -i $MEE_PATH -o $PROCESSED_DATA_PATH --lang english --split_path $SPLIT_PATH --split split3
python process_mee.py -i $MEE_PATH -o $PROCESSED_DATA_PATH --lang english --split_path $SPLIT_PATH --split split4
python process_mee.py -i $MEE_PATH -o $PROCESSED_DATA_PATH --lang english --split_path $SPLIT_PATH --split split5


