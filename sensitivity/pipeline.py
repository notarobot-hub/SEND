"""
Use perturbation.py and most_sensitive.py to analyze sensitivity.
"""
from run_with_temperature import *
from perturbation import *

if __name__ == '__main__':

    INPUT = "Obesity hypoventilation syndrome ( also known as Pickwickian syndrome ) is a condition in which severely overweight people fail to breathe rapidly enough or deeply enough , resulting in low blood oxygen levels and high blood carbon dioxide ( CO2 ) levels . Many people with this condition also frequently stop breathing altogether for short periods of time during sleep ( obstructive sleep apnea ) , resulting in many partial awakenings during the night , which leads to continual sleepiness during "

    # Load the model and tokenizer
    accelerator = Accelerator()
    model_name = "EleutherAI/pythia-70m"
    embeddings = run_with_temperature(accelerator, model_name, INPUT)
    # start a new process for explain_features_multiprocess function with input embeddings
    most_effective = explain_features_multiprocess(embeddings)
    print(most_effective)

