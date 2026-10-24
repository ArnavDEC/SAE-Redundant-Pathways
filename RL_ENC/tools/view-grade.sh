#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Wrong number of arguments. $0 expects 1 argument, a container id"
  exit 1
fi
output=$(docker cp $1:/grader/grade.json - | tar -xO)
if echo $output | grep "backtrace" &> /dev/null; then
    echo "$output"| jq -r .backtrace | sed 's/\\n/\n/g'
else
    echo "$output"
    echo
fi
