export MAVEN_PATH="./"
export PROCESSED_DATA_PATH="../../processed_data/maven"

python process_maven.py -i $MAVEN_PATH -o $PROCESSED_DATA_PATH --split split1
python process_maven.py -i $MAVEN_PATH -o $PROCESSED_DATA_PATH --split split2
python process_maven.py -i $MAVEN_PATH -o $PROCESSED_DATA_PATH --split split3
python process_maven.py -i $MAVEN_PATH -o $PROCESSED_DATA_PATH --split split4
python process_maven.py -i $MAVEN_PATH -o $PROCESSED_DATA_PATH --split split5
