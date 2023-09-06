from array import array
from scipy import sparse
import numpy as np
import json
import pandas as pd
import shap
from aix360.metrics import faithfulness_metric
from tqdm import tqdm
from time import sleep
import pickle

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from supervised_tf_idf import SupervisedTermWeights


class HateXplainExperiments:
    
    def __init__(self, description):
        self.description = description

    def prepare_dataset(self):

        # Pre-processing
        # Filter out sentences where annotators did not agree on the labels
        # Select as final label the label where the majority of the annotators agreed on
        # The rationale of annotator c is not kept as often not available

        f = open('dataset.json')
        data = json.load(f)

        cleaned_dataset = []

        for k,v in data.items():
            
            individual_data = {}
            individual_data['post_tokens'] = v['post_tokens']
            individual_data['post_tokens_joined'] = ' '.join(v['post_tokens'])
            individual_data['annotator_a_label'] = v['annotators'][0]['label']
            individual_data['annotator_b_label'] = v['annotators'][1]['label']
            individual_data['annotator_c_label'] = v['annotators'][2]['label']
            
            labels = [v['annotators'][0]['label'], v['annotators'][1]['label'], v['annotators'][2]['label']]
            final_label = max(labels, key=labels.count)
            individual_data['final_label'] = final_label
            
            if set(labels) != 3:       
                if len(v['rationales']) == 0: 
                    individual_data['annotator_a_rationale'] = []
                    individual_data['annotator_b_rationale'] = []
                    individual_data['annotator_c_rationale'] = []
                    individual_data['final_rationale'] = [0 for token in v['post_tokens']]                    
                else:  
                    individual_data['annotator_a_rationale'] = v['rationales'][0]
                    individual_data['annotator_b_rationale'] = v['rationales'][1]
                
                    if final_label != v['annotators'][1]['label']:
                        individual_data['final_rationale'] = v['rationales'][0]
                    else:
                        individual_data['final_rationale'] = v['rationales'][1]
                            
                cleaned_dataset.append(individual_data)
        
        dataset = pd.DataFrame(cleaned_dataset)
        return dataset
    
    def prepare_properties(self, dataset):

        train_indices, test_indices = train_test_split(dataset.index, test_size=0.2, random_state=42)
        
        self.df_train = dataset.loc[train_indices]
        self.df_test = dataset.loc[test_indices]
        
        self.X_train = self.df_train['post_tokens_joined']
        self.y_train = self.df_train['final_label']    
        self. X_test = self.df_test['post_tokens_joined']
        self.y_test = self.df_test['final_label']
        
        # self.vectorizer = TfidfVectorizer(norm='l2')
        # self.X_train_vect = self.vectorizer.fit_transform(self.X_train, self.y_train)
        # self.X_test_vect = self.vectorizer.transform(self.X_test)

        self.vectorizer = TfidfVectorizer(norm='l2')
        self.X_train_vect = self.vectorizer.fit_transform(self.X_train, self.y_train)
        self.X_test_vect = self.vectorizer.transform(self.X_test)
              
        self.train_tfidf_matrix = self.X_train_vect.toarray()
        self.test_tfidf_matrix = self.X_test_vect.toarray()

        self.train_rationales = self.df_train['final_rationale']
        self.test_rationales = self.df_test['final_rationale']
        self.train_sentences = self.df_train['post_tokens']
        self.test_sentences = self.df_test['post_tokens']

    def preprocess_training_data_option_one(self): 

        # Exponential function applied to tokens in the TF-IDF vectorised training matrix tagged 
        # as not indicative in the human rationale [a]

        train_sentences = self.train_sentences
        train_rationales = self.train_rationales
        vectorizer = self.vectorizer
        train_tfidf_matrix = self.train_tfidf_matrix

        for enum, (sentence, global_rationale) in enumerate(zip(train_sentences, train_rationales)):
            previously_executed_tokens = []
            for token, rationale in zip(sentence, global_rationale):
                if token not in previously_executed_tokens and token in vectorizer.vocabulary_ and rationale == 0:
                        token_idx = vectorizer.vocabulary_[token]
                        token_tfidf_value = train_tfidf_matrix[enum][token_idx]
                        train_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1
                        previously_executed_tokens.append(token)

        X_train_vect = sparse.csr_matrix(train_tfidf_matrix)
        return X_train_vect

    def preprocess_training_data_option_two(self):

        train_sentences = self.train_sentences
        train_rationales = self.train_rationales
        vectorizer = self.vectorizer
        train_tfidf_matrix = self.train_tfidf_matrix

        # Exponential function applied to tokens in the TF-IDF vectorised training matrix 
        # (1) tagged as not indicative in offensive and hatespeech related human rationales and 
        # (2) and not unique to sentences labelled as normal [b]

        tokens_in_positive_class = []
        for sentence, global_rationale in zip(train_sentences, train_rationales):
            if any(i == 1 for i in global_rationale): #this means the observation is offensive or contains hatespeech
                for token in sentence:
                    tokens_in_positive_class.append(token)
        unique_tokens_in_positive_class = list(set(tokens_in_positive_class))

        for enum, (sentence, global_rationale) in enumerate(zip(train_sentences, train_rationales)): 
            previously_executed_tokens = []
            if any(i == 1 for i in global_rationale): # (1)
                for token, rationale in zip(sentence, global_rationale):
                    if token not in previously_executed_tokens and token in vectorizer.vocabulary_ and rationale == 0:
                            token_idx = vectorizer.vocabulary_[token]
                            token_tfidf_value = train_tfidf_matrix[enum][token_idx]
                            train_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1
                            previously_executed_tokens.append(token)
            else: # (2)
                for token in sentence:
                    if token not in unique_tokens_in_positive_class and token not in previously_executed_tokens and token in vectorizer.vocabulary_: 
                        token_idx = vectorizer.vocabulary_[token]
                        token_tfidf_value = train_tfidf_matrix[enum][token_idx]
                        train_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1
                        previously_executed_tokens.append(token)
        
        X_train_vect = sparse.csr_matrix(train_tfidf_matrix)
        return X_train_vect

    def preprocess_training_data_option_three(self):

        train_sentences = self.train_sentences
        train_rationales = self.train_rationales
        vectorizer = self.vectorizer
        train_tfidf_matrix = self.train_tfidf_matrix

        # (1) Assign np.exp() * 0.5 to tokens found indicative in offensive and hatespeech tagged sentences
        # (2) Leave tf-idf value of tokens found indicative in offensive and hatespeech tagged sentences untouched
        # (3) Assign 2 ** n - 1 to tokens in normal tagged sentences    

        for enum, (sentence, global_rationale) in enumerate(zip(train_sentences, train_rationales)):
            
            if any(i == 1 for i in global_rationale): #this means the observation is offensive or contains hatespeech        
                previously_executed_tokens = []
                previously_executed_indicative_tokens = []
                for token, rationale in zip(sentence, global_rationale):
                    
                    if token in vectorizer.vocabulary_ and token not in previously_executed_tokens and rationale == 1:
                        token_idx = vectorizer.vocabulary_[token]
                        token_tfidf_value = train_tfidf_matrix[enum][token_idx]
                        train_tfidf_matrix[enum][token_idx] = np.exp(token_tfidf_value) * 0.5 # (1)
                        previously_executed_tokens.append(token)
                        previously_executed_indicative_tokens.append(token)
                    
                    elif token in vectorizer.vocabulary_ and token not in previously_executed_indicative_tokens and rationale == 0:
                        token_idx = vectorizer.vocabulary_[token]
                        token_tfidf_value = train_tfidf_matrix[enum][token_idx]
                        train_tfidf_matrix[enum][token_idx] = token_tfidf_value # (2)
                        previously_executed_tokens.append(token)

            else:
                previously_executed_tokens = []
                for token, rationale in zip(sentence, global_rationale):
                    if token in vectorizer.vocabulary_ and token not in previously_executed_tokens:
                        token_idx = vectorizer.vocabulary_[token]
                        token_tfidf_value = train_tfidf_matrix[enum][token_idx] 
                        train_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1 # (3)
                        previously_executed_tokens.append(token)
        
        X_train_vect = sparse.csr_matrix(train_tfidf_matrix)
        return X_train_vect

    def preprocess_test_data_option_one(self):

        train_sentences = self.train_sentences
        test_sentences = self.test_sentences
        train_rationales = self.train_rationales
        vectorizer = self.vectorizer
        test_tfidf_matrix = self.test_tfidf_matrix
 
        # Exponential function applied to tokens in the TF-IDF vectorised test matrix never tagged 
        # as indicative in offensive or hatespeech related human rationales from the training set [c]

        indicative_tokens_in_positive_class = []
        for sentence, global_rationale in zip(train_sentences, train_rationales):
            if any(i == 1 for i in global_rationale): #this means the observation is offensive or contains hatespeech
                for token, rationale in zip(sentence, global_rationale):
                    if rationale == 1:
                        indicative_tokens_in_positive_class.append(token)
    
        unique_indicative_tokens_in_positive_class = list(set(indicative_tokens_in_positive_class))
        
        for sentence, global_rationale in zip(train_sentences, train_rationales):
            if all(i == 0 for i in global_rationale): #this means the observation is normal
                for token in sentence:
                    if token in unique_indicative_tokens_in_positive_class:
                        unique_indicative_tokens_in_positive_class.remove(token)

        for enum, sentence in enumerate(test_sentences):
            previously_executed_tokens = []
            for token in sentence:
                if token not in unique_indicative_tokens_in_positive_class and token not in previously_executed_tokens and token in vectorizer.vocabulary_:  
                    token_idx = vectorizer.vocabulary_[token]
                    token_tfidf_value = test_tfidf_matrix[enum][token_idx]
                    test_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1
                    previously_executed_tokens.append(token)

        X_test_vect = sparse.csr_matrix(test_tfidf_matrix)

        return X_test_vect


    def preprocess_test_data_option_two(self):
        
        # Exponential function applied to tokens in the TF-IDF vectorised test matrix never tagged 
        # as indicative in the human rationales from the training set

        train_sentences = self.train_sentences
        test_sentences = self.test_sentences
        train_rationales = self.train_rationales
        vectorizer = self.vectorizer
        test_tfidf_matrix = self.test_tfidf_matrix

        indicative_tokens_in_positive_class = []
        for sentence, global_rationale in zip(train_sentences, train_rationales):
            if any(i == 1 for i in global_rationale): #this means the observation is offensive or contains hatespeech
                for token, rationale in zip(sentence, global_rationale):
                    if rationale == 1:
                        indicative_tokens_in_positive_class.append(token)
        unique_indicative_tokens_in_positive_class = list(set(indicative_tokens_in_positive_class))

        for enum, sentence in enumerate(test_sentences):
            previously_executed_tokens = []
            for token in sentence:
                if token not in unique_indicative_tokens_in_positive_class and token not in previously_executed_tokens and token in vectorizer.vocabulary_:  
                    token_idx = vectorizer.vocabulary_[token]
                    token_tfidf_value = test_tfidf_matrix[enum][token_idx]
                    test_tfidf_matrix[enum][token_idx] =  2 ** token_tfidf_value - 1
                    previously_executed_tokens.append(token)

        X_test_vect = sparse.csr_matrix(test_tfidf_matrix)
        return X_test_vect

    def train_model(self, X_train_vect, y_train):

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))
        
        estimators = [
            ('sdg', SGDClassifier(penalty='l2', fit_intercept=True, loss='log_loss')),
            ('nb', ComplementNB()),
            # ('dt', DecisionTreeClassifier(max_depth=3))
        ]

        sclf = StackingClassifier(estimators=estimators, passthrough=True, final_estimator=SGDClassifier(
            max_iter=100000,
            loss='log_loss', 
            penalty='l2', 
            n_jobs=-1,
            fit_intercept=True)
        )

        params = {
            'nb__alpha': [0.0001, 0.001, 0.01, 0.1, 2],
            'nb__norm': [False, True],
            'nb__class_prior': [None, [0.307048, 0.405944, 0.287008]], 
            'sdg__max_iter': [100000],
            'sdg__average': [True, False],
            'sdg__alpha': [0.0001, 0.001, 0.01, 0.1],
            'sdg__learning_rate': ['optimal'],
            'sdg__class_weight': ['balanced', class_weights],
            # 'dt__min_samples_split': [2, 3, 4, 5, 6],
            # 'dt__min_samples_leaf': [1, 2, 3, 4, 5, 6],
            # 'dt__criterion': ['gini', 'entropy'],
        }

        # pipe = Pipeline([
        #     ('clf', 'passthrough')
        # ])

        # params = [  
        #     {
        #         'clf': [SVC(kernel='linear', probability=True)],
        #         'clf__C': [0.5, 1, 2, 5],
        #         'clf__max_iter': [100000],
        #         'clf__class_weight': ['balanced', {
        #         'hatespeech': class_weights[0],
        #         'normal': class_weights[1],
        #         'offensive': class_weights[2]
        # }],
        #     }
        # ]

        search = GridSearchCV(estimator=sclf, param_grid=params, scoring='accuracy', cv=StratifiedKFold(5, random_state=43, shuffle=True), refit=True, n_jobs=-1)
        search.fit(X_train_vect, y_train)
        
        return search

    def evaluate_faithfulness(self, best_algorithm, vectorizer, X_train_vect, X_test_vect, test_sentences, test_rationales):

        explainer = shap.KernelExplainer(best_algorithm.predict_proba, shap.sample(X_train_vect, 10), model_output='probability')
        shap_values = explainer.shap_values(X_test_vect)
        
        base = []
        features = vectorizer.vocabulary_.keys()
        for sentence, global_rationale in zip(test_sentences, test_rationales):
            individual_base = np.zeros(len(features))
            if any(i == 1 for i in global_rationale):
                for token, rationale in zip(sentence, global_rationale):
                    if rationale == 1 and token in vectorizer.vocabulary_:
                        idx = vectorizer.vocabulary_.get(token)
                        individual_base[idx] = 1
            base.append(individual_base) 

        labels = {
            'hatespeech': 0,
            'normal': 1,
            'offensive': 2,
        }
        
        ncases = len(test_sentences)
        fait = np.zeros(ncases)

        for i in tqdm(range(ncases)):
            predicted_class = best_algorithm.predict(X_test_vect[i])[0]
            shap_idx = labels[predicted_class]
            fait[i] = faithfulness_metric(best_algorithm, X_test_vect.toarray()[i], shap_values[shap_idx][i], base[i])

        return np.mean(fait), np.std(fait)


    def get_performance_metrics(self, X_train_vect=None, X_test_vect=None):

        # kbest = SelectKBest(chi2, k=11000)
        # X_train_vect = kbest.fit_transform(X_train_vect, self.y_train)
        # X_test_vect = kbest.transform(self.X_test_vect)

        description = self.description
        y_train = self.y_train
        y_test = self.y_test
        # test_sentences = self.test_sentences
        # test_rationales = self.test_sentences
        # vectorizer = self.vectorizer

        if X_test_vect == None:
            X_test_vect = self.X_test_vect

        if X_train_vect == None:
            X_train_vect = self.X_train_vect
        
        search = self.train_model(X_train_vect, y_train)

        best_algorithm = search.best_estimator_

        filename = 'best_algorithm.sav'
        pickle.dump(best_algorithm, open(filename, 'wb'))

        best_params = search.best_params_
        print(best_params)
        probs = best_algorithm.predict_proba(X_test_vect)
        y_pred = best_algorithm.predict(X_test_vect)
        
        cv_score = search.best_score_
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # mean_fait, std_fait = self.evaluate_faithfulness(best_algorithm, vectorizer, X_train_vect, X_test_vect, test_sentences, test_rationales)
        
        data = {
            'Description': [description],
            'CV score': [cv_score],
            'Accuracy': [accuracy],
            'Balanced accuracy': [balanced_accuracy],
            'AUC-ROC score': [roc_auc],
            'F1 score': [f1],
            'Precision': [precision],
            'Recall': [recall],
            # 'Faithfulness mean': [mean_fait],
            # 'Faithfulness std': [std_fait],
        }

        print(data)

        metrics = pd.DataFrame(data)
        with open('metrics.csv', 'a') as f:
            metrics.to_csv(f, header=f.tell()==0)


if __name__ == '__main__':

    # description = "SVM: Exponential function applied to tokens in the TF-IDF vectorised training matrix tagged as not indicative in the human rationale [a]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_one()
    # hatexplain.get_performance_metrics(X_train_vect)

    # ---

    # description = "Exponential function applied to tokens in the TF-IDF vectorised training matrix (1) tagged as not indicative in offensive and hate-speech related human rationales and (2) and not unique to sentences labelled as normal [b]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_two()
    # hatexplain.get_performance_metrics(X_train_vect)

    # ---

    # description = "[a] + Exponential function applied to tokens in the TF-IDF vectorised test matrix never tagged as indicative in offensive or hate speech related human rationales from the training set [c]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_one()
    # X_test_vect = hatexplain.preprocess_test_data_option_one()
    # hatexplain.get_performance_metrics(X_train_vect, X_test_vect)

    # ---

    # description = "[a] + Exponential function applied to tokens in the TF-IDF vectorised test matrix never tagged as indicative in the human rationales from the training set [d]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_one()
    # X_test_vect = hatexplain.preprocess_test_data_option_two()
    # hatexplain.get_performance_metrics(X_train_vect, X_test_vect)

    # ---

    # description = "[b] + [c]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_two()
    # X_test_vect = hatexplain.preprocess_test_data_option_one()
    # hatexplain.get_performance_metrics(X_train_vect, X_test_vect)

    # ---

    # description = "[b] + [d]"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_two()
    # X_test_vect = hatexplain.preprocess_test_data_option_two()
    # hatexplain.get_performance_metrics(X_train_vect, X_test_vect)

    # ---
    
    # description = "[a] with 50% max components in TF-IDF"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_one()
    # hatexplain.get_performance_metrics(X_train_vect)

    # ---

    # description = "[b] with 50% max components in TF-IDF"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_two()
    # hatexplain.get_performance_metrics(X_train_vect)

    #---
    
    # description = "[a] with SupervisedTermWeights"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_one()
    # hatexplain.get_performance_metrics(X_train_vect)

    #---

    # description = "[b] with SupervisedTermWeights"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # X_train_vect = hatexplain.preprocess_training_data_option_two()
    # hatexplain.get_performance_metrics(X_train_vect)

    #---

    # description = "No injection"
    # hatexplain = HateXplainExperiments(description)
    # dataset = hatexplain.prepare_dataset()
    # hatexplain.prepare_properties(dataset)
    # hatexplain.get_performance_metrics()