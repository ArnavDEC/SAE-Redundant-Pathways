#!/usr/bin/env bash

if [ $# -gt 1 ]; then
    echo "Wrong number of arguments; expects a rollout folder or no arguments"
    exit 1
fi

ROLLOUT_DIR=${1:-runs/$(ls -t runs | head -n 1)}

if [ -e rollouts ]; then
    while true; do
        read -p "You already directory called rollouts, do you want to overwrite it?" yn
        case $yn in
            [Yy]* ) echo "Overwiting existing rollout dir"; rm -r rollouts; break;;
            [Nn]* ) exit 1;;
            * ) echo "Please answer yes or no";;
        esac
    done
fi

if [ -f rollouts.tar.gz ]; then
    while true; do
        read -p "You already have a file named rollouts.tar.gz, do you want to overwrite it?" yn
        case $yn in
            [Yy]* ) echo "Overwiting existing rollout dir"; rm -r rollouts.tar.gz; break;;
            [Nn]* ) exit 1;;
            * ) echo "Please answer yes or no";;
        esac
    done
fi

mkdir -p rollouts

for container in $(cat $ROLLOUT_DIR); do
    for log in grader engine entrypoint; do
        docker cp $container:/logs/${log}.log rollouts/${container}-${log}.log
    done
done
tar czf rollouts.tar.gz rollouts
rm -r rollouts
echo "Your rollout archive is available in rollouts.tar.gz"
