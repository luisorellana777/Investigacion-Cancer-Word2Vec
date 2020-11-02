#api= https://biopython.org/DIST/docs/api/Bio.Entrez-module.html
#ejemplos url = https://www.ncbi.nlm.nih.gov/books/NBK25499/
from Bio import Entrez
import pandas as pd
import re  # For preprocessing
from time import time  # To time our operations
import spacy  # For preprocessing

Entrez.email = 'your_email@provider.com'

#Bases de datos = https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/ 
#Obtener lista de id por tema y periodo de tiempo
#https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&reldate=99999&datetype=edat&retmax=10&usehistory=y
#Sinonimos: https://pubchem.ncbi.nlm.nih.gov/compound/Oxime-_-methoxy-phenyl#section=Synonyms&fullscreen=true

#crear lista de palabras claves a buscar y agregar todos los ids encontrados a la lista "pmids"

key_words = list()
key_words.append("oxime-methoxy-phenyl")
key_words.append("SCHEMBL8530447")
key_words.append("HUYDCTLGGLCUTE-HJWRWDBZSA-N")
key_words.append("Methyl N-hydroxybenzenecarboximidoate")
key_words.append("methyl (z)-N-hydroxybenzenecarboximidate")
key_words.append("Methyl")
key_words.append("hydroxybenzenecarboximidoate")
key_words.append("hydroxybenzenecarboximidate")
key_words.append("methoxy")
key_words.append("Oxime")
key_words.append("phenyl")

#key_words.append("Curcumin")
#key_words.append("Diferuloylmethane")
#key_words.append("Turmeric Yellow")
#key_words.append("Turmeric")

pmids =  list()

for key_word in key_words:
    handle = Entrez.esearch(db="pubmed", retmax=9999999, term=key_word)
    records = Entrez.read(handle)
    # Obtener abstract por id 
    if len(records['IdList']) > 0:
        pmids.extend(records['IdList'])

pmids = (list(pmids))
print(pmids)


handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pmids)), retmode="json", rettype="xml")

records = Entrez.read(handle)

abstracts = list()
pmids =  list()

for pubmed_article in records['PubmedArticle']:
    try:
        abstracts.append(pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
        pmids.append(pubmed_article['MedlineCitation']['PMID'])
    except Exception:
        #abstracts.append("-")
        #pmids.append("-")
        pass

df = pd.DataFrame({"Ids":pmids, "Abstract":abstracts})

df.to_csv (r'D:\Investigacion Word2Vec\Modelo\pubmed_abstracts.csv', index = None, header=True)


print(df)

############LIMPIEZA DE LOS DATOS   

df.isnull().sum()
df = df.dropna().reset_index(drop=True)
df.isnull().sum()

nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

brief_cleaning = (re.sub("<..........>|<.........>|<........>|<.......>|<......>|<.....>|<....>|<...>|<..>|<.>|[^A-Za-z]+", ' ', str(row)).lower() for row in df['Abstract'])

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))


df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape

df_clean.to_csv (r'D:\Investigacion Word2Vec\Modelo\pubmed_sentences.csv', index = None, header=True)


handle.close()