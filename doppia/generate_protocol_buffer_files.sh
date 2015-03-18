#!/bin/sh

set -x  # will print excecuted commands

echo "Generating objects detection files..."
cd src/objects_detection/
protoc --cpp_out=./ detector_model.proto detections.proto
protoc --python_out=../../tools/objects_detection/ detector_model.proto detections.proto
cd ../..

cd src/helpers/data
protoc --cpp_out=./ DataSequenceHeader.proto
protoc --python_out=../../../tools/data_sequence DataSequenceHeader.proto
cd ../../..

echo "End of game. Have a nice day!"
