import re
import sys
from utils import write_status
from nltk.stem.porter import PorterStemmer


porter_stemmer = PorterStemmer()
def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(comment):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    comment = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', comment)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    comment = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', comment)
    # Love -- <3, :*
    comment = re.sub(r'(<3|:\*)', ' EMO_POS ', comment)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    comment = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', comment)
    # Sad -- :-(, : (, :(, ):, )-:
    comment = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', comment)
    # Cry -- :,(, :'(, :"(
    comment = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', comment)
    return comment


def preprocess_comment(comment):
    processed_comment = []
    # Convert to lower case
    comment = comment.lower()
   
    # Replace 2+ dots with space
    comment = re.sub(r'\.{2,}', ' ', comment)
    # Strip space, " and ' from comment
    comment = comment.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    comment = handle_emojis(comment)
    # Replace multiple spaces with a single space
    comment = re.sub(r'\s+', ' ', comment)
    words = comment.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            word = str(porter_stemmer.stem(word))
            processed_comment.append(word)

    return ' '.join(processed_comment)


def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            tweet = line
            processed_tweet = preprocess_comment(tweet)
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))
            write_status(i + 1, total)
    save_to_file.close()
    
    return processed_file_name

