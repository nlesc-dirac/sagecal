#!/usr/bin/env bash

version=$(date +"%d%m%Y")

for folder in \
	sl7-test \
	ubuntu1604-test

do
    name=${folder%%/}
    echo
    echo "==========================================="
    echo "Building: "$name
    echo "==========================================="

    cmd="docker build -t fdiblen/$name:$version -t fdiblen/$name:latest ./$name"
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

    #docker push fdiblen/$name:$version
    #docker push fdiblen/$name:latest
done
