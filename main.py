import torch
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from nltk import pos_tag
import nltk
from pathlib import Path
import spacy
# pip install pattern should work
from pattern.en import conjugate, PARTICIPLE, referenced, INDEFINITE, pluralize

template_repo = './templates/'
single_templates= 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'

data_repo = './data/'
test_data = 'test2.txt' #@ SJ made

#-----------------------------------------

# 1. DirectTemplate
regex = '[A-Z][^A-Z]*'
def apply_template1(relation, head, tail):
    template = " ".join(re.findall(regex, relation))
    return ' '.join([head, template, tail])

#test
apply_template1('IsA','baseball','sport')

# 2. PredefinedTemplate, single template, grammar = False
template_loc= template_repo+'relation_map.json'
grammar=False
language_model=None
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
grammar = grammar
with open(template_loc, 'r') as f:
    templates = json.load(f)

def clean_text(words):
    new_words = words.split(' ')
    doc = nlp(words)
    first_word_POS = doc[0].pos_
    if first_word_POS == 'VERB':
        new_words[0] = conjugate(new_words[0], tense=PARTICIPLE)
    if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
        if new_words[0] != 'a' or new_words[0] != 'an':
            new_words[0] = referenced(new_words[0])
    elif first_word_POS == 'NUM' and len(new_words) > 1:
        new_words[1] = pluralize(new_words[1])
    return ' '.join(new_words)

def apply_template2(relation, head, tail):
    if grammar:
        head = clean_text(head)
        tail = clean_text(tail)
    sent = templates[relation].format(head, tail) #may be problematic because of multiple templates@
    return sent

#test
apply_template2('MadeOf','bottle','plastic') #works


# 3. PredefinedTemplate, single template, grammar = True
grammar = True

#test
apply_template2('UsedFor','pen','write') #works

# 4. EnumeratedTemplate, multiple templates

language_model=None
template_loc='./templates/relation_map_multiple.json'
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def formats(phrase):
    doc = nlp(phrase)
    first_word_POS = doc[0].pos_
    tokens = phrase.split(' ')
    new_tokens = tokens.copy()
    new_phrases = []
    # original
    new_phrases.append(' '.join(new_tokens))
    # with indefinite article
    if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
        new_tokens[0] = referenced(tokens[0])
        new_phrases.append(' '.join(new_tokens))
    # with definite article
    if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
        new_tokens[0] = "the "+tokens[0]
        new_phrases.append(' '.join(new_tokens))
    # as gerund
    if first_word_POS == 'VERB':
        new_tokens[0] = conjugate(tokens[0], tense=PARTICIPLE)
        new_phrases.append(' '.join(new_tokens))
        if len(tokens) > 1:
            if tokens[1] == 'to' and len(tokens) > 2:
                new_tokens[2] = referenced(tokens[2])
            else:
                new_tokens[1] = referenced(tokens[1])
        new_phrases.append(' '.join(new_tokens))
        new_tokens[0] = tokens[0]
        new_phrases.append(' '.join(new_tokens))
    # account for numbers
    if first_word_POS == 'NUM' and len(tokens) > 1:
        new_tokens[1] = pluralize(tokens[1])
        new_phrases.append(' '.join(new_tokens))
    return new_phrases

with open(template_loc, 'r') as f:
    templates = json.load(f)

def get_candidates(relation, head, tail):
    heads = formats(head)
    tails = formats(tail)
    templates2 = templates[relation]
    candidate_sents = []
    for h in heads:
        for t in tails:
            for temp in templates2:
                candidate_sents.append((temp.format(h, t)))
    return candidate_sents

def apply_template4(relation, head, tail):
    candidate_sents = get_candidates(relation, head, tail)
    return candidate_sents
    #sent, head, tail = get_best_candidate(candidate_sents)
    #return sent

#test
apply_template4('HasProperty','basketball player','tall') #works

# 5. EnumeratedTemplate, multiple templates, select the best sentence
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'
template_loc='./templates/relation_map_multiple.json'
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
enc = GPT2Tokenizer.from_pretrained('gpt2') 
gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)
language_model=gpt
model = language_model
if model is not None:
    model.eval()
    #model.to(device) #this step is necessary so far, as device is difficult to define

def formats(phrase):
    doc = nlp(phrase)
    first_word_POS = doc[0].pos_
    tokens = phrase.split(' ')
    new_tokens = tokens.copy()
    new_phrases = []
    # original
    new_phrases.append(' '.join(new_tokens))
    # with indefinite article
    if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
        new_tokens[0] = referenced(tokens[0])
        new_phrases.append(' '.join(new_tokens))
    # with definite article
    if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
        new_tokens[0] = "the "+tokens[0]
        new_phrases.append(' '.join(new_tokens))
    # as gerund
    if first_word_POS == 'VERB':
        new_tokens[0] = conjugate(tokens[0], tense=PARTICIPLE)
        new_phrases.append(' '.join(new_tokens))
        if len(tokens) > 1:
            if tokens[1] == 'to' and len(tokens) > 2:
                new_tokens[2] = referenced(tokens[2])
            else:
                new_tokens[1] = referenced(tokens[1])
        new_phrases.append(' '.join(new_tokens))
        new_tokens[0] = tokens[0]
        new_phrases.append(' '.join(new_tokens))
    # account for numbers
    if first_word_POS == 'NUM' and len(tokens) > 1:
        new_tokens[1] = pluralize(tokens[1])
        new_phrases.append(' '.join(new_tokens))
    return new_phrases

with open(template_loc, 'r') as f:
    templates = json.load(f)

def get_candidates(relation, head, tail):
    heads = formats(head)
    tails = formats(tail)
    templates2 = templates[relation]
    candidate_sents = []
    for h in heads:
        for t in tails:
            for temp in templates2:
                candidate_sents.append((temp.format(h, t)))
    return candidate_sents

def score_sent(candidate):
    sent = candidate
    sent = ". "+sent
    try:
        tokens = enc.encode(sent)
    except KeyError:
        return 0
    #context = torch.tensor(tokens, dtype=torch.long, device=device).reshape(1,-1)
    context = torch.tensor(tokens, dtype=torch.long).reshape(1,-1)
    logits, _ = model(context) #@
    log_probs = logits.log_softmax(2)
    sentence_log_prob = 0
    for idx, c in enumerate(tokens):
        if idx > 0:
            sentence_log_prob += log_probs[0, idx-1, c]
    return sentence_log_prob.item() / (len(tokens)**0.2)

def get_best_candidate(candidate_sents):
    candidate_sents.sort(key=score_sent, reverse=True)
    return candidate_sents[0]

def apply_template5(relation, head, tail):
    candidate_sents = get_candidates(relation, head, tail)
    sent = get_best_candidate(candidate_sents)
    return sent

#test
apply_template5('HasProperty','basketball player','tall') #works
apply_template5('MadeOf','book','paper') #works


# 6. Process the conceptnet test data (remove those with 0)
# 6.2 Batch processing

test_data = 'test3.txt' #@ SJ made
# Load tuples
tuple_dir = data_repo + test_data
sens = []
with open(tuple_dir) as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        #print(row[0],row[1],row[2])
        sen = apply_template5(row[0],row[1],row[2])
        print(sen)
        sens.append(sen)

# Save to file
f = open(sent_dir, 'w')
for ele in sens:
    f.write(ele+'\n')
f.close() #output is 1 sentence per row in sent.txt


# =============== code factory
# Save to file method 1
sent_dir = data_repo + 'sent.txt'
with open(sent_dir, 'a') as the_file:
    the_file.write(str(sens)) #output is a list of strings in sent.txt

