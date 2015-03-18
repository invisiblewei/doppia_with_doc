#!/bin/sh
zippath="AFW.zip"
imgpath="testimages" 

cd "data/afw"

if [ ! -e $zippath ]; then 
    echo "wget"
    wget http://www.ics.uci.edu/~xzhu/face/AFW.zip
fi 
if [ ! -x $imgpath ]; then 
    echo "unzip"
    unzip AFW
fi 

rm -rf __*

echo "get afw finished"

