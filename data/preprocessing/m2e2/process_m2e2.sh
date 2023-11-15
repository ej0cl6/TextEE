export M2E2_PATH="./m2e2_annotations"
export PROCESSED_DATA_PATH="../../processed_data/m2e2"

python process_m2e2.py -i $M2E2_PATH -o $PROCESSED_DATA_PATH --split split1
python process_m2e2.py -i $M2E2_PATH -o $PROCESSED_DATA_PATH --split split2
python process_m2e2.py -i $M2E2_PATH -o $PROCESSED_DATA_PATH --split split3
python process_m2e2.py -i $M2E2_PATH -o $PROCESSED_DATA_PATH --split split4
python process_m2e2.py -i $M2E2_PATH -o $PROCESSED_DATA_PATH --split split5
