# Vul-Tracker

## Vul-Tracker: Automated Vulnerability Fixes with Deep Co-Matching Learning

Python library dependencies:
+ tensorflow -v : 1.13.1
+ numpy -v : 1.18.5
+ nltk -v : 3.4.5
+ flask -v : 1.1.1
+ GitHub-Flask -v : 3.2.0
+ gensim -v : 3.8.3
+ scipy -v : 1.4.1 
+ others: sklearn, bs4,

---

Dataset:

[VulFixMiner]: Reference paper: Zhou et al. "Finding A Needle in a Haystack: Automated Mining of Silent Vulnerability Fixes. ACM, 2021. (link: [[http://yuyue.github.io/res/paper/DupPR-msr2017.pdf](https://ieeexplore.ieee.org/abstract/document/9678720)])
<including a total of 256 Java and Python projects.>

---

#### **If you want to successfully run Vul-Tracker, please set up the environment as follows.**

1. Ensure at least two graphics cards with 8GB of memory each to avoid memory overflow issues.

2. Set the character encoding to UTF-8.
   
3. codebert-base needs to be downloaded to your local machine in advance.

4. Import the corresponding library versions according to the library dependencies.
   
---        

#### **After completing the above settings, you can run the Vul-Tracker code following the steps below.**
       #### **Extract seven levels of code features.**
             + related_commit_feature.py
  
                `python related_line_feature.py`
    
               This will generate patch_variant_1_finetune_1_epoch_best_model and the embeddings for related-commit-level code changes.
            + related_file_feature.py
  
               `python related_file_feature.py`
    
               This will generate patch_variant_2_finetune_1_epoch_best_model and the embeddings for related-file-level code changes.
           + related_hunke_feature.py
  
              `python related_hunk_feature.py`
    
              This will generate patch_variant_3_finetune_1_epoch_best_model and the embeddings for related-hunk-level code changes.
          
           + unrelated_commit_feature.py
  
              `python unrelated_commit_feature.py`
    
              This will generate patch_variant_5_finetune_1_epoch_best_model and the embeddings for unrelated-commit-level code changes.
           + unrelated_file_feature.py
  
              `python unrelated_file_feature.py`
    
              This will generate patch_variant_6_finetune_1_epoch_best_model and the embeddings for unrelated-file-level code changes.
           + unrelated_hunk_feature.py
  
              `python unrelated_hunk_feature.py`
    
              This will generate patch_variant_7_finetune_1_epoch_best_model and the embeddings for unrelated-hunk-level code changes.
          + unrelated_line_feature.py
  
              `python unrelated_line_feature.py`
    
              This will generate patch_variant_8_finetune_1_epoch_best_model and the embeddings for unrelated-line-level code changes.

     #### **Run the Trusted Label Loss Correction Component, Dynamic Contextual Attention Component, and Context Matching Component to obtain the experimental results.**
          + python classifier_hat.py
          + python adjustment_runner.py

#### **After the above processing, you can obtain the evaluation results of the model.**
      #### **RQ1:**
         + python evaluator.py --rq 1

     #### **RQ2:**
         + python evaluator.py --rq 2

    #### **RQ3**
         + python classifier_xiao.py
         + python adjustment_runner.py
         + python evaluator.py --rq 3
    #### **RQ4**
         + python classifier_cnn.py  python adjustment_runner.py  python evaluator.py --rq 1
         + python classifier_lstm.py  python adjustment_runner.py  python evaluator.py --rq 1
         + python classifier_rnn.py  python adjustment_runner.py  python evaluator.py --rq 1 
