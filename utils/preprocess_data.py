import re


def get_punctuation_signs_for_prediction():
    # The text used in this exercise appear to use the double dash "--" as a
    # hyphen. Therefore, double dashes should be considered as punctuation
    # sign but not simple dashes.
    return [",", ";", ":", "!", "?", ".", "'", '"', "(", ")"]


def get_punctuation_signs_for_tokenization():
    """
    same as method above - but adds 5 additional punctuation signs - to be used
    at tokenization time (and NOT for the predictions)
    """
    return [",", ";", ":", "!", "?", ".", "'", '"', "(", ")",
            "[", "]", "{", "}", "..."]


def get_punctuation_marker():
    return "<punctuation>" 
            

def group_lines_into_paragraphs(lines):
    # Group individual lines into paragraphs (paragraphs are assumed to be
    # separated by empty lines)
    paragraphs = []
    lines_in_current_paragraph = []

    for l in lines:
        # Remove double spaces, leading spaces and trailing spaces
        l = l.strip()

        if len(l) == 0:
            # Wrap up current paragraph and setup for next one
            current_paragraph = " ".join(lines_in_current_paragraph)
            paragraphs.append(current_paragraph)
            lines_in_current_paragraph = []
        else:
            # Add current line to the current paragraph
            lines_in_current_paragraph.append(l)
    
    # Wrap up last paragraph
    current_paragraph = " ".join(lines_in_current_paragraph)
    paragraphs.append(current_paragraph)
    
    return paragraphs
    

def pad_around_substrings(text, substrings):
    for substr in substrings:
        text = text.replace(substr, " " + substr + " ")
    return text    
   
    
def pad_punctuation(text):
    """
    Separate a given text into a list of tokens where a token is either a word
    or a punctuation sign.

    """
    
    # Add spaces around all the punctuation signs (except the apostrophe)
    punctuation_signs = get_punctuation_signs_for_tokenization()
    text = pad_around_substrings(text,
                                 [p for p in punctuation_signs if p != "'"])

    # Only pad around apostrophes that are used as quotation marks, not
    # as part of a word (ex: "you're happy"). If an apostrophe is both
    # preceded and followed by a letter, it is assumed to be part of a word.
    # Pad around apostrophes where at least one surrounding character is not
    # a letter
    if "'" in punctuation_signs:
        text = re.sub("\'[^A-Za-z]", " ' ", text)
        text = re.sub("[^A-Za-z]\'", " ' ", text)
               
    # Remove duplicate space characters
    text = re.sub(" +", " ", text)
    
    # Reassemble any ellipsis that may have been split up
    text = text.replace(". . .", "...")
    
    # Remove any leading and trailing whitespace we might have created
    return text.strip()
    
    
def get_input_label_from_text(text):
    punctuation_signs = get_punctuation_signs_for_prediction()
    punctuation_marker = get_punctuation_marker()                    
                         
    input = []
    target = []
    
    for word in text.split():
        if word in punctuation_signs:
            input.append(punctuation_marker)
            target.append(word)
        else:
            input.append(word)

    return input, target
    

def get_text_from_input_label(input, label):
    punctuation_marker = get_punctuation_marker()
    
    label_idx = 0
    reconstruction = []
    
    for word in input:
        if word == punctuation_marker:
            reconstruction.append(label[label_idx])
            label_idx += 1
        else:
            reconstruction.append(word)
            
    return " ".join(reconstruction)

def build_vocab(inputs):
    """
    Build a vocabulary from the input sentences.
    The vocabulary is a mapping from words to unique integer IDs.
    inputs: are in form of a list of sentences, where each sentence is a list of words.
    ['<punctuation>', 'now', '<punctuation>', 'it', 'is', 'a', 'fact', '<punctuation>', 'gentlemen', '<punctuation>']
    """
    vocab = {}
    special_tokens = ["<pad>", "<unk>", "<punctuation>"]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for tokens in inputs:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def preprocess_file(filename):

    # Recover file content
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Group individual lines into paragraphs
    paragraphs = group_lines_into_paragraphs(lines)
    
    # Remove chapter headers (they are simply roman numerals followed by a ".")
    chapter_headers = [("%s.") % s
                       for s in ["I", "II", "III", "IV", "V", "VI", "VII"]]
    paragraphs = [p for p in paragraphs if p not in chapter_headers]
    
    # Transform all letters to lowercase (the presence of uppercase characters
    # would be a strong clue that a period should be used)
    paragraphs = [p.lower() for p in paragraphs]
    
    # Ensure the present of one space character between punctuation signs
    # and between words and punctuation signs.
    paragraphs = [pad_punctuation(p) for p in paragraphs]
    
    # From every paragraph, extract the input and corresponding target
    inputs, targets = zip(*[get_input_label_from_text(p) for p in paragraphs])
    
    """
    # Validate the decomposition
    new_paragraphs = [get_text_from_input_label(inp, tar) for inp, tar in zip(inputs, targets)]
    for p, new_p in zip(paragraphs, new_paragraphs):
        assert (p == new_p)
    """
    
    return inputs, targets

if __name__ == '__main__':   

    # build the vocabulary and save it to a file, as well as the label2id mapping
    import json
    import os
    from glob import glob

    import yaml

    #load config
    with open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    input_files = glob(os.path.join(config["TRAIN_DIR"], "*.txt"))
    
    all_inputs = []
    all_labels = []
    for file in input_files:
        inputs, labels = preprocess_file(file)
        all_inputs.extend(inputs)
        all_labels.extend(labels)
    vocab = build_vocab(all_inputs)
    with open(config["VOCAB_PATH"], "w") as f:
        json.dump(vocab, f, indent=4)   

    labels = enumerate(get_punctuation_signs_for_prediction())
    label2id = {label: idx for idx, label in labels}
    with open(config["LABEL2ID_PATH"], "w") as f:
        json.dump(label2id, f, indent=4)

    #also id2label
    id2label = {idx: label for label, idx in label2id.items()}
    with open(config["ID2LABEL_PATH"], "w") as f:
        json.dump(id2label, f, indent=4)

    print(f"Vocabulary and label2id saved to {config['VOCAB_PATH']} and {config['LABEL2ID_PATH']}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Label2id size: {len(label2id)}")

    text = "Hello, world! This is a test. Let's see how it works."
    print("Original text:", text)
    text = pad_punctuation(text)
    print("Padded text:", text)
    inputs, targets = get_input_label_from_text(text)
    print("Inputs:", inputs)
    print("Targets:", targets)
    reconstructed_text = get_text_from_input_label(inputs, targets)
    print("Reconstructed text:", reconstructed_text)
    assert text == reconstructed_text, "Reconstructed text does not match original text"
