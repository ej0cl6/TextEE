export PHEE_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/phee"

python process_phee.py -i $PHEE_PATH -o $PROCESSED_DATA_PATH --split split1
python process_phee.py -i $PHEE_PATH -o $PROCESSED_DATA_PATH --split split2
python process_phee.py -i $PHEE_PATH -o $PROCESSED_DATA_PATH --split split3
python process_phee.py -i $PHEE_PATH -o $PROCESSED_DATA_PATH --split split4
python process_phee.py -i $PHEE_PATH -o $PROCESSED_DATA_PATH --split split5
