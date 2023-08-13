# Extracting Causal Phrases from Scientific Texts
The idea here is to extract causal phrases present in the abstracts of scientific articles. Causal phrases are in a way claims made by the authors of the scientific articles. The reason to use the abstract instead of full paper is beacuse abstract tends be the condensed version of the paper, and focuses on the contributions/claims of that paper. And, it is computationally easier to process. I am using SpaCy package to train the model, and use it to identify the causal phrases in the abstracts.   

# Data
## Model training
Here, I have used the dataset from (here)[https://github.com/tanfiona/CausalNewsCorpus]. The link has multiple dataset related to causal phrases, events, and so on. But, for my project, I have specifically used the subtask-1 dataset in the version 2 folder. If you are using this data, please cite the orginal authors. 

## Scientific Articles
The abstracts I downloaded are from Scopus database. Particularly, I focused on the journal called "Transport Policy." The abstracts of all papers that are published in Transport Policy journal from year 2017 to 2022. Among these papers, I kept only the papers that are related to transportation planning.   
