import pandas as pd
import re
import subprocess,os
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier
import treetaggerwrapper


current_path=os.path.abspath(os.getcwd())
#authorize executable file on linux environments
subprocess.check_call(['chmod', '+x', current_path+"/treetagger/treetagger/bin/tree-tagger"])
class Data:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = open(self.data_file, "r", encoding="utf-8")
    
    def split_data(self):
        data = self.data.readlines() #list of lines
        titles= list(set([x for x in data if x.startswith("***")]))
        delimiters = list(set([x for x in data if x.startswith("---")]))
        return data, titles, delimiters

class useful:
    def tupling(y, N=2):
        result=[y[n:n+N] for n in range(0, len(y), N)]
        return result
    def striping(text):
       text=text.replace("-", "").replace("*", "").strip()
       return text
    def structuring(text, bornes):
        regex_pattern = "|".join([re.escape(b) for b in bornes])
        text=re.split(f'({regex_pattern})', "".join(text))
        return [x for x in text if len(x)>1]
    def hypehnation(text):
        text=text.replace("\n", " ")
        text=text.replace("- ", "")
        text=re.sub(r'\([^()]*\)', '', text)
        return text.strip()
    def dictionarization(text, bornes):
        regex_pattern = "|".join([re.escape(b) for b in bornes])
        text=re.split(f'({regex_pattern})', "".join(text))
        #usar tupling
        return {useful.striping(x[0]):x[1] for x in useful.tupling([x for x in text if len(x)>1], 2)}
    def grafias(text):
        forggiven_characters=[("ę", "e"), ("æ", "e"), ("œ", "e"), ("<", ""), ("(", ""), (")", ""), ("\xa0", ""),
                     (">", ""), ("*", ""), ("°", "_"), ("[", ""), ("]", ""), ("«", ""), ("»", ""), 
                     (",", " ,"), (".", " ."), (";", " ;"), ("  ", " "), (" | | ", " "), (" | |", ""), ("\n", ""), ("\t", ""),
                     ("V", "U"), ("v", "u"), ("J", "I"), ("j", "i")]
        for grafia in forggiven_characters:
            text=str(text).replace(grafia[0], grafia[1])
        return text
    def bio_conll_single(pr_sentence):
        tokenized_text=[token.text for token in pr_sentence]
        conll_text=["O"]*len(tokenized_text)
        for x in pr_sentence.get_spans('ner'):
            if len(x)==1:
                try:
                    conll_text[x[0].idx-1]="B-"+x.tag
                except:
                    print(x[0].idx, x)
            else:
                conll_text[x[0].idx-1]="B-"+x.tag
                for token in x[1:]:
                    conll_text[token.idx-1]="I-"+x.tag
        
        return conll_text
        

class transformation(Data):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.data, self.titles, self.delimiters = self.split_data()
    
    def transform(self):
        self.titles = useful.dictionarization(self.data, self.titles)
        self.titles = {k:useful.dictionarization(v,self.delimiters) for k,v in self.titles.items()}
        return self.titles
    
class TAL_features_extraction(transformation):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.titles = self.transform()
        self.titles= {k:[self.discourse_parts["act"]] for k,v in self.titles.items()} #call discourse parts function here and add it to the dictionary
        #now call lematizer on each second element of each value of the dictionary
        self.titles= {k:[v[0], self.named_entities(self.lematizer(v[1]))] for k,v in self.titles.items()} #call lematizer and NER function here and add it to the dictionary
        return self.titles

    def discourse_parts(sentence):
        DIS_model = SequenceTagger.load('models/discours_parts_05_02_2022.pt')
        DIS_sentence= Sentence(sentence)
        DIS_model.predict(DIS_sentence)
        tokenized_text=[str(token).split("Token: ")[1].split()[1] for token in DIS_sentence]

        parts_discours=[]
        for x in DIS_sentence.get_spans('ner'):
            index=" ".join([(str(y).split("Token: ")[1]) for y in x]).split()[::2]#captura los index
            index=[int(index[0])-1, int(index[-1])]
            part=str(x).split("[− Labels: ")[1].replace("]", "")
            parts_discours.append([part, " ".join(tokenized_text[index[0]:index[1]])])
        return parts_discours
    
    def lematizer(sentence):
        tagger=treetaggerwrapper.TreeTagger(TAGLANG="la", TAGDIR="treetagger/treetagger")

        if len(sentence)>1:
            #tags_cap=tagger.tag_text(sentence) #keep the original text
            #tags_cap=[x.split("\t")[0] for x in tags_cap]

            tags=tagger.tag_text(useful.grafias(sentence).lower())# pass lowerized text to treetagger as that produce more accurate pos and lemma
            tags=[x.split("\t") for x in tags]
        return tags


    def named_entities(tags):
        FLAT_model=SequenceTagger.load('models/best-model_flat_13_03_2022.pt')
        FLAT_sentence=Sentence([x[0] for x in tags])
        FLAT_model.predict(FLAT_sentence)
        entities=useful.bio_conll_single(FLAT_sentence)

        tags=[[x[0], x[1], x[2].split("_", 1)[0], entities[i], kk] for i,x in enumerate(tags)] #combining treetagger and Flair results
        return tags

    


class TAL_to_HTML(TAL_features_extraction):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.data, self.titles, self.delimiters = self.split_data()
        

actas=TAL_features_extraction("Saint_Hubert_full_text.txt")
print(actas.keys())

actas=transformation("Saint_Hubert_full_text.txt").transform()
for k,v in actas.items():
    print(k)
    for kk,vv in v.items():
        if kk=="act":
            vv=useful.hypehnation(vv)
            vv=TAL_features_extraction.discourse_parts(vv)
            for p in vv:
                print(p[0], TAL_features_extraction.lematizer(p[1]))
            print("\n\n")
        
        '''
        self.data = pd.DataFrame(self.data, columns=["title", "content"])
        self.data["title"] = self.data["title"].apply(lambda x: x.replace("---", ""))
        self.data["content"] = self.data["content"].apply(lambda x: x.replace("---", ""))
        self.data["title"] = self.data["title"].apply(lambda x: x.replace("
        new=transformation("Saint_Hubert_full_text.txt").transform()
for k,v in new.items():
    print(k)
    for kk,vv in v.items():
        print(kk, useful.hypehnation(vv))
    print("\n\n")
        '''
