#!/bin/bash

for d in dataset/*/; do
  echo -n $d": ";
  (ls "$d" | wc -l);
done
