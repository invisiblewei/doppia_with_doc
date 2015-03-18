#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a trained model, this script will plot a visual representation of
the channels features usage.
The plot should be comparable to figure 3 in
Dollar et al. "Integral Channel Features" BMVC 2009,
but here we plot the model, not the specific activation for a single image
"""

import os.path
import sys
filedir = os.path.join(os.path.dirname(__file__))
sys.path.append(os.path.join(filedir, ".."))
sys.path.append(os.path.join(filedir, "..", "..", "helpers"))
import detector_model_pb2
from optparse import OptionParser
from google.protobuf import text_format

def main():

    parser = OptionParser()
    parser.description = \
        "Reads a trained detector model and plot its content"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="FILE", type="string",
                      help="path to the model file")
    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
    else:
        parser.error("'input' option is required to run this program")

    model_filename = options.input_path
    #model = read_model(model_filename)
    
    model = detector_model_pb2.DetectorModelsBundle()
    f = open(model_filename, "rb")
    model.ParseFromString(f.read())
    f.close()
    

    s = text_format.MessageToString(model)
    print(s)

    return


if __name__ == '__main__':
    main()

# end of file
