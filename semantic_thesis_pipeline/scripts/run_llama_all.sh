#!/bin/bash

python run_analysis.py --model llama-7b --stage full
python run_analysis.py --model llama-2-7b --stage full
python run_analysis.py --model llama-3-8b --stage full
python run_analysis.py --model llama-3.1-8b --stage full
