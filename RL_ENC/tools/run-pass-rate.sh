#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Wrong number of arguments. $0 expects 1 argument, a container id"
  exit 1
fi

for f in `cat $1`; do ./tools/view-grade.sh $f | jq .score ; done | jq -sn 'inputs | add/length'
