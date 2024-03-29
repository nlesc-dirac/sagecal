#!/usr/bin/env bash

version=$(date +"%d%m%Y")

for folder in \
	ubuntu2004-*

do
    name=${folder%%/}
    echo
    echo "==========================================="
    echo "Building: "$name
    echo "==========================================="

    cmd="docker build -t $name:$version ./$name"
    echo
    echo "Running:"
    echo $cmd
    echo
    echo
    eval $cmd

    if [ $? -eq 0 ]
    then
      echo
      echo "Successfully built the $name image!"
    else
      echo
      echo "Could not build the $name image" >&2
      exit $?
    fi

done
