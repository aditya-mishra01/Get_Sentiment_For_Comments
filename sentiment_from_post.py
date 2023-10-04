import sys
from find_comment import get_comments
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from lstm import process_comments
import numpy as np
if __name__ == '__main__':
    model_path=""
    model=load_model(model_path)
    comments=get_comments(sys.argv[0])
    comments,_=process_comments(comments)
    comments=pad_sequences(comments, maxlen=40, padding='post')
    prediction=model.predict(comments, batch_size=128, verbose=1)
    p=np.round(prediction).count(0)
    n=np.round(prediction).count(1)

    print(f"This post has {p} postive comments and {n} negative comments")

