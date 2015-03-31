headhunter.txt is too big to send in email.
headhunter_baseline is the version with 5 components in the paper.

The structure of the model is defined in detector_model.proto

You can transform proto.bin into txt with following steps:
1. If you have install python, install the protobuf for python.
2. run $ python bin2txt.py  -i headhunter_baseline.proto.bin > headhunter_baseline.proto.txt