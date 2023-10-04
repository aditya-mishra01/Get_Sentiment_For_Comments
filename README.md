# Get_Sentiment_For_Comments
Deep learning Model to get Sentiment Analysis for Facebook and Instagram Post Comments
# For Training
First Pre-process your csv dataset using preprocess.py , it will generate a new csv file \
Generate frequency distribution for unigrams and bi grams using get_freq.py , it will generate 2 pkl files \
Glove-File : GloVe words vectors from StanfordNLP which match our dataset for seeding word embeddings.\
Use lstm.py to train and save your model . DON'T FORGET TO CHANGE THE FILE PATHS in lstm.py 
# For Running 
Change the model_path in sentiment_from_post.py
Run using command python <sentiment_from_post.py {url}> where url is link to the post 


Dataset I used for training : https://www.kaggle.com/datasets/subhajeetdas/hate-comment

