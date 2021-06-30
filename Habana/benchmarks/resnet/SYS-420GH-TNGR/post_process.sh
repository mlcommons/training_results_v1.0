#!/bin/bash

SRC=$1
DST=$2
NAMESPACE=worker0

if [ -z $2 ]; then
  echo "Usage\n$0 input output"
  exit 1
fi
grep MLLOG $SRC |grep $NAMESPACE| grep -v mlp_log.py | sed 's#^.*stdout>:##' > $DST
