"""
Use perturbation.py and most_sensitive.py to analyze sensitivity.
This file wil load the sensitivities of the LLM and give us the eigenscore change with the removed sensitive and random neurons
"""
from pyexpat import model
from run_with_temperature import *
from perturbation import *
from most_sensitive import *
from delta_vector_across_checkpoints import efficient_extract_sensitive
import json
import pandas as pd


if __name__ == "__main__":

    sub_diff = []
    mult_diff = []
    sense_overall = []
    random_overall = []
    eigen_overall = []

    model_size = 'pythia-1b'
    model_name = "EleutherAI/" + model_size
    top_k_percent = efficient_extract_sensitive(model_size)
    top_k_num = 10

    count = 0

    with open(f'.././data/hidden_layer_activations_model_30_to_43_{model_size}_hallu.json') as f:
        data = json.load(f)

        df = pd.DataFrame(data)

        for text_num, text in enumerate(df['text']):
            

            warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")
            GUIDANCE = "You are my completion assistant. Continue the following prompt and complete it based on facts:  "
            # INPUT = "Obesity hypoventilation syndrome ( also known as Pickwickian syndrome ) is a condition in which severely overweight people fail to breathe rapidly enough or deeply enough , resulting in low blood oxygen levels and high blood carbon dioxide ( CO2 ) levels . Many people with this condition also frequently stop breathing altogether for short periods of time during sleep ( obstructive sleep apnea ) , resulting in many partial awakenings during the night , which leads to continual sleepiness during "

            INPUT = GUIDANCE + text

            # Load the model and tokenizer
            accelerator = Accelerator()
            revision_eigenscores = []

            for revision in range(133000, 144000, 1000):

                # Run 10 inferences on the fully trained model
                embeddings = run_with_temperature(accelerator, model_name, INPUT, REVISION=revision)
                # start a new process for explain_features_multiprocess function with input embeddings
                # most_effective_indices_on_eigenscore = explain_features_multiprocess(embeddings)    # get the neuron effectiveness scores dict


            
                """full_emb = extract_embeddings(accelerator, model_name, INPUT)
                SLT_emb = get_SLT_embedding(full_emb)
                net = combined_sensitivity(SLT_emb)
                top_k_percent = top_k_percent_sensitive(net, 1)  # get the top 5% most sensitive neurons as a dictionary
                print(f'Top K Percent Len: {len(top_k_percent)}')
                
                most_effective_indices_on_eigenscore = explain_sensitive_vs_random(embeddings, top_k_percent)
                
                print(f"Most effective neurons on eigenscore: {most_effective_indices_on_eigenscore}")
                print(f"Most sensitive neurons shape: {len(top_k_percent)}")
                """



                set_start_method('spawn', force=True)
                
                eigen = compute_eigenscore(cp.array(embeddings))
                
                revision_eigenscores.append(eigen.get())
                                
                if count >= 80:
                    break
                    
                count += 1

            eigen_overall.append(revision_eigenscores)
            print(eigen)

        diff = np.diff(eigen_overall, axis=0)
        
        for i, eigen in enumerate(eigen_overall):
            j = i*1000 + 133000
            plt.plot(diff, label=f"Revision {i} to {i + 1000}")

        plt.title("Difference in EigenScore Between Checkpoints")
        plt.ylabel("Difference from one checkpoint to next")
        plt.savefig(f"eigen_score_fluctuation_text_{text_num}")
