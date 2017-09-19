import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    probabilities = []
    guesses = []
    all_Xlengths = test_set.get_all_Xlengths()
    all_sequences = test_set.get_all_sequences()
    for sequence in all_sequences:
        best_guess = 0
        best_score = float("-inf")
        probability = {}
        X, lengths = all_Xlengths[sequence]
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except:
                logL = float("-inf")
            probability[word] = logL
            if logL > best_score:
                best_score = logL
                best_guess = word
        probabilities.append(probability)
        guesses.append(best_guess)
        
    return probabilities, guesses
