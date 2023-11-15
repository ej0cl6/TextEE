export ACE05_PATH="./ace_2005_td_v7/data/English"
export SPLIT_PATH="./split-en"
export FINAL_OUTPUT_PATH="../../processed_data/ace05-en/"

python process_ace05_en.py -i $ACE05_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split1
python process_ace05_en.py -i $ACE05_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split2
python process_ace05_en.py -i $ACE05_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split3
python process_ace05_en.py -i $ACE05_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split4
python process_ace05_en.py -i $ACE05_PATH -o $FINAL_OUTPUT_PATH --sent_map sent_map.json --token_map token_map.json --lang english --split_path $SPLIT_PATH --split split5


