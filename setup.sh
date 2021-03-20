#! /bin/bash

pip install -r requirements.dev.txt
pip install -r requirements.txt
pre-commit install
