# -*- coding: utf-8 -*-

import os
import sys
import re
import nltk
from ner_feature_extraction import NERFeatureExtractor
from ner_recognition import NERTagger
from optparse import OptionParser
import uuid

def load_config(config_file):
    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            param, value = line.split()
            if param == 'with_pos':
                value = bool(int(value))
            config[param] = value
    return config

def cross_validate(ner, training_file, string_file_uid, remove_temp_file=True):

    tmpdir, crfpath, feature_scheme = '', '', ''
    for key in ner:
	if key == 'tmpdir':
	    tmpdir = ner[key]
	if key == 'crfpath':
	    crfpath = ner[key]
	if key == 'feature_scheme':
	    feature_scheme = ner[key]

    train_crfsuite_file = os.path.join(tmpdir, str(uuid.uuid1()) + '.crfsuite')
    feature_ext = NERFeatureExtractor(feature_scheme=feature_scheme)
    feature_ext.extract_features(training_file, train_crfsuite_file, 
                                 feature_scheme=feature_scheme)

    # Evaluation 
    comd = '%s learn -g5 -x %s ' % (crfpath, train_crfsuite_file) + '> ' + string_file_uid
    print(comd)
    os.system(comd)

    # remove .crfsuite file
    if remove_temp_file:
        os.remove(train_crfsuite_file)

def removeSpecialChar(strings):
	stringss = strings.encode('utf-8').translate(None, '!@#$"\\:(),?-')
	return stringss

def Read_file(string_file):
    list_string = []
    # Load file
    with open(string_file, "r") as files:
	content = files.readlines()
	for row in content:    
	    row = unicode(row, 'utf-8')
	    list_string.append(row)
    return list_string

def Load_val(string_file):
    holdout, iteration, list_iteration = [], [], []
    # load file
    list_str = Read_file(string_file)
    # Process
    for s in range(len(list_str)):
	text_split = list_str[s].strip().split(' ')
	if len(text_split)!=1:
	    if text_split[0] == 'Macro-average':
		precision = float(removeSpecialChar(text_split[4]))
	        iteration.append(precision)
		recall = float(removeSpecialChar(text_split[5]))
	        iteration.append(recall)
	 	f1 = float(removeSpecialChar(text_split[6]))
	        iteration.append(f1)
	    if text_split[0] == 'Item' and text_split[1]== 'accuracy:':
		#print 'it s here '
		acc = float(removeSpecialChar(text_split[5]))
		iteration.append(acc)
	    if text_split[1] == 'Iteration' and iteration!=[]:
	        list_iteration.append(iteration)
	        iteration = []
	    if text_split[0] == 'Holdout' and text_split[2]!='1' and list_iteration!=[] or s==(len(list_str)-2):
	        holdout.append(list_iteration)
		list_iteration = []
    return holdout    

def Statistic(list_holdout):
    count, precis_total, recall_total, f1_total, accuracy_total = 0, 0, 0, 0, 0
    for hold in list_holdout:
	count = count + 1
	precis, recal, f1_score, accuracy = 0.0, 0.0, 0.0, 0.0
	for itera in hold:
	    #if len(itera)!=0:
	    precis = precis + itera[0]
	    recal = recal + itera[1]
	    f1_score = f1_score + itera[2]
	    accuracy = accuracy + itera[3]
	precis_total = precis_total + (precis / len(hold))
	recall_total = recall_total + (recal / len(hold))
	f1_total = f1_total + (f1_score / len(hold))
	accuracy_total = accuracy_total + (accuracy / len(hold))
	print 'score precision of holdout group ' + str(count) + ': '+ str((precis / len(hold)) * 100) + '  %'
	print 'score recall of holdout group ' + str(count) + ': '+ str((recal / len(hold)) * 100) + '  %'
	print 'score F1 of holdout group ' + str(count) + ': '+ str((f1_score / len(hold)) * 100) + '  %'
	print 'score Accuracy of holdout group ' + str(count) + ': '+ str((accuracy / len(hold)) * 100) + '  %'
	print '----------------------------' 
    print '\n'
    print '----- Score total ----------'
    print ' Precision score: ' + str(precis_total / len(list_holdout) * 100) + '  %'
    print ' Recall score: ' + str(recall_total / len(list_holdout) * 100) + '  %'
    print ' F1 score: ' + str(f1_total / len(list_holdout) * 100) + '  %'
    print ' accuracy score: ' + str(accuracy_total / len(list_holdout) * 100) + '  %'
    print '\n'

if __name__ == '__main__':
    training_file = './data/largeData.edata'
    config = load_config('./config.conll.txt')
    string_file_ui = str(uuid.uuid4())
    cross_validate(config, training_file, string_file_ui)    
    statistic_dt = Load_val(string_file_ui)   
    Statistic(statistic_dt)
    # remove file 
    os.remove(string_file_ui)
