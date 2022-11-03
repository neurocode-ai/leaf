#!/bin/bash

# Recursively remove the Cython compiled
# __pycache__ directories from root ".",
# use with care! Invoking rm -rf on all
# directories with name *pycache*.
function traverse() {
  for file in "$1"/*
  do
    if [ ! -d "${file}" ] ; then
      continue
    else
      if [[ "${file}" == *"pycache"* ]] ; then
        rm -rf "${file}"
      else
        traverse "${file}"
      fi
    fi
  done
}

printf "Removing __pycache__/ recursively..."
traverse "."
printf "OK\n"
