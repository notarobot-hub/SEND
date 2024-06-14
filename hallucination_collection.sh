#!/bin/bash

model_names=()
model_sizes=()
model_checkpoints=()

cd "HaluEval/evaluation"

# This for loop will collect all the hallucinated examples in the dataset
for name in ${model_names[@]}
do
    for size in ${model_sizes[@]}
    do
        for check in ${model_checkpoints[@]}
        do
            # Figure out how to access the individual model checkpoints
            eval_model=""
            
            # This will save two np arrays corresponding to the responses of the model
            python evaluate.py --task qa --model eval_model
        done
    done
done
