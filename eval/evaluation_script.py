import random

import numpy

from preprocessing.preprocess_data import (
    get_punctuation_marker, get_punctuation_signs_for_prediction,
    get_punctuation_signs_for_tokenization, preprocess_file)


class Model:
    """
    This is the interface that your model wrapper should expose
    """
    
    def predict(self, text):
        """
        Return the model's prediction on a given example.
        
        This function expects a list of strings (words) as input and returns
        a list of strings (punctuation signs as output).
        
        Every string element in 'text' is either a word or a marker indicating
        that a punctuation sign is missing at this position. The value of this
        marker ("<punctuation>") can be obtained by calling the function
        'get_punctuation_marker()'.
        
        The output should be a list of strings containing valid punctuation
        signs. This list should contain as many elements as the number of times
        that the punctuation marker occurred in 'text'. The list of valid
        punctuation signs can be obtained by calling the function
        'get_punctuation_signs_for_prediction()'.
        
        Example of input and possible valid output :
        text : ["yesterday", "<punctuation>", "i", "went", "to", "the",
                "park", "<punctuation>", "tomorrow", "<punctuation>" "i'll",
                "stay", "home", "<punctuation>"]
        output : [",", ".", ",", "!"]
        """
        raise NotImplementedError()
        

class RandomModel(Model):
    """
    Sample implementation of a model which makes random predictions
    """

    def __init__(self):
        self.punctuation_signs = get_punctuation_signs_for_prediction()
        self.punctuation_marker = get_punctuation_marker()
    
    def predict(self, text):
        # Read the input and, every time we see a missing punctuation sign,
        # we add a random punctuation sign to our prediction.
        prediction = []
        for word in text:
            if word == self.punctuation_marker:
                prediction.append(random.choice(self.punctuation_signs))
        return prediction


def evaluate_model(model, inputs, labels):

    # Obtain the model's confusion matrix
    classes = get_punctuation_signs_for_prediction()
    conf_matrix = numpy.zeros((len(classes), len(classes)), dtype="int32")
    for input, label in zip(inputs, labels):
        
        # Get the model's prediction
        prediction = model.predict(input)
 
        # Ensure that the predictions are valid
        assert len(prediction) == len(label), f"Invalid number of predictions  pred: {len(prediction)}, gold: {len(label)} for input: {input}"
        assert all(isinstance(p, str) for p in prediction), "Model predicted non-string punctuation signs: {}".format(prediction)
        for p in prediction:
            assert p in classes, "Model predicted an invalid punctuation sign: {}".format(p)
            
        # Populate the confusion matrix
        for p, l in zip(prediction, label):
            conf_matrix[classes.index(l), classes.index(p)] += 1

    # Compute the generalized F1 score from the confusion matrix
    class_f1_scores = []
    for i, punctuation in enumerate(classes):
        precision = (conf_matrix[i, i] + 1e-6) / (conf_matrix[:, i].sum() + 1e-6)
        recall = (conf_matrix[i, i] + 1e-6) / (conf_matrix[i, :].sum() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        class_f1_scores.append(f1)

    return sum(class_f1_scores) / len(class_f1_scores)


if __name__ == '__main__':   

    # Load and process the data
    inputs1, labels1 = preprocess_file("a_scandal_in_bohemia.txt")
    inputs2, labels2 = preprocess_file("the_red_headed_league.txt")
    inputs3, labels3 = preprocess_file("a_case_of_identity.txt")
    test_inputs = inputs1 + inputs2 + inputs3
    test_labels = labels1 + labels2 + labels3
    
    # Load the model
    # START : Modify this section
    from eval.eval_wrapper import LSTMWrapper
    model = LSTMWrapper()
    # END: do not modify below
    
    # Obtain the model's confusion matrix
    avg_f1_score = evaluate_model(model, test_inputs, test_labels)
    print("Average F1 score : {}".format(avg_f1_score))
