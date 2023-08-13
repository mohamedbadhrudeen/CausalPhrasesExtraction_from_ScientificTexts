# Extracting Causal Phrases from Scientific Texts
The idea here is to extract causal sentences present in the abstracts of scientific articles. Causal phrases are in a way claims made by the authors of the scientific articles. The reason to use the abstract instead of full paper is beacuse abstract tends be the condensed version of the paper, and focuses on the contributions/claims of that paper. And, it is computationally easier to process. I am using SpaCy package to train the model, and use it to identify the causal phrases in the abstracts. The model is essentially a text categorization model: given a sentence, the model will predict if that sentence has causal phrase(s) or not.   

# Data
## Model training
Here, I have used the dataset from [here](https://github.com/tanfiona/CausalNewsCorpus). The link has multiple dataset related to causal phrases, events, and so on. But, for my project, I have specifically used the subtask-1 dataset in the version 2 folder. If you are using this data, please cite the orginal authors. 

## Scientific Articles
The abstracts I downloaded are from Scopus database. Particularly, I focused on the journal called "Transport Policy." The abstracts of all papers that are published in Transport Policy journal from year 2017 to 2022. Among these papers, I kept only the papers that are related to transportation planning.   

# Spacy Training
I did not include the trained model as the size of the model was over 500 mb. However, you can use the config.cfg, causal_training_data.spacy, causal_deve_data.spacy to train your own model. causal_training_data.spacy and causal_dev_data.spacy are training and dev data converted into SpaCy's format. The training your custom model process is very simple to do. Follow the steps below.

1. Open the terminal (I used Conda).
2. Type the following command: python -m spacy train config.cfg --output ./output --paths.train ./causal_training_data.spacy --paths.dev ./causal_dev_data.spacy
3. Make sure the path is correct. If the command gives you any error, try giving the full path for causal_training_data.spacy and causal_dev_data.spacy.
4. The above command will produce the trained model in the --output path you give.
5. Load that model in the CausalPhrasesExtraction.py.

You will notice, I used both trained model and SpaCy's default model. It was because, the trained model did not have 'sentencizer', to split the abstracts into multiple sentences. So, I had to use the default SpaCy's model 'en_core_web_lg'.

# End Use
You can use the extracted causal sentences to look for claims/topics related to transportation planning. Here, I have used a simple dimensionality reduction techniques (PCA, Isomap, TSNE) to see how these causal sentences are distributed in relation to others. 

# Questions
If you have any questions, or suggestions please contact me. You can get my details [here](https://mohamedbadhrudeen.github.io/).
