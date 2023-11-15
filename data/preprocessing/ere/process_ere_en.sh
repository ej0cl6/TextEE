export ERE_EN_PATH="./DEFT_English_Light_and_Rich_ERE_Annotation/data/"
export SPLIT_PATH="./split-en"
export FINAL_OUTPUT_PATH="../../processed_data/richere-en/"

python process_ere_en.py -i $ERE_EN_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split1
python process_ere_en.py -i $ERE_EN_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split2
python process_ere_en.py -i $ERE_EN_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split3
python process_ere_en.py -i $ERE_EN_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split4
python process_ere_en.py -i $ERE_EN_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split5
