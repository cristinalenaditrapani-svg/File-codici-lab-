import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Download pacchetti
nltk.download("punkt_tab")
nltk.download("stopwords")

# 2. Caricamento Dataset
dati = pd.read_csv("dataset_comments_bodycam.csv", encoding="utf-8")

list(dati.columns)

#PULIZIA DATASET
# 3. Rimozione stopwords del dataset grezzo 
stop_words_eng = set(stopwords.words("english"))

# Rimozione avverbi comuni 
parole_avverbi = {"also", "clearly", "may", "yet", "said", "told", "knew", "like", "would", "could", "get", "immediately", "second","even", "really", "much", "well"}
stop_words_eng.update(parole_avverbi)

# 4. Dizionario sinonimi
mappa_sostituzioni = {
    "gabbie": "gabby", "gaby": "gabby", "gabbi": "gabby", "gabrielle": "gabby", 
    "petito": "gabby", "gabbys": "gabby", "fiancé": "gabby", "fiancée": "gabby", "gp": "gabby",
    "fiance": "gabby",
    "brians":"brian", "laundrie": "brian", "bryan": "brian", "brain": "brian", "bl": "brian", "boyriend":"brian",
    "cops": "police", "cop": "police", "officer": "police", "officers": "police", "policeman": "police" 
    }


# 5. Pulizia testo
def preprocessa_testo (testo) -> list:

    # RIMUOVERE I LINK E GLI AT
    testo = re.sub(r"http\S+", "", testo) 
    testo = re.sub("@\S+", "", testo)
    testo = re.sub("[^A-Za-z\à\è\é\ì\ò\ù]+", " ", testo)
    
    parole_token = nltk.word_tokenize(testo)
    parole_token = [n.lower().strip() for n in parole_token]
    
    # RISOLTO BUG LISTA VUOTA: usiamo un nome nuovo ("token_puliti")
    token_puliti = []
    for w in parole_token:
        # LA RICERCA ISTANTANEA: usa il set 'stop_words_eng'
        if w not in stop_words_eng and len(w) > 1:
            # Traduce il sinonimo o lascia la parola originale
            w_finale = mappa_sostituzioni.get(w, w)
            token_puliti.append(w_finale)
            
    return token_puliti
    
dati["testi_puliti"]=dati.comments.apply(lambda x: preprocessa_testo(x))


print(dati["testi_puliti"])

dati['testi_puliti'].head() 
dati['testi_puliti'].tail()


#Creazione lista 
corpus_tokenizzato = dati["testi_puliti"].tolist()

num_documenti = len(corpus_tokenizzato)
num_token_totali = sum(len(doc) for doc in corpus_tokenizzato)
vocab_grezzo = set(w for doc in corpus_tokenizzato for w in doc)

print(f"Documenti: {num_documenti:,}")
print(f"Token totali: {num_token_totali:,}")
print(f"Vocabolario grezzo: {len(vocab_grezzo):,} parole uniche")


#TABELLA CORPUS
dati_metrica = {
    "Metrica del Dataset": ["Commenti", "Token Totali", "Vocabolario Unico"],
    "Valore Assoluto": [num_documenti, num_token_totali, len(vocab_grezzo)],
    }
    
df_statistiche = pd.DataFrame(dati_metrica)

df_visualizzazione = df_statistiche.copy()
df_visualizzazione["Valore Assoluto"] = df_visualizzazione["Valore Assoluto"].map('{:,}'.format)

print("\n TABELLA DESCRITTIVA DATASET")
print(df_visualizzazione.to_string(index=False))

# Esportazione tabella 
df_statistiche.to_csv("tabella_metodologia_corpus.csv", index=False)
df_statistiche.to_excel("tabella_metodologia_corpus.xlsx", index=False)

print("\nTabella salvata nei file 'tabella_metodologia_corpus.csv' e '.xlsx'!")


# Filtro di sicurezza: scarto dei commenti svuotati dalla pulizia
corpus_tokenizzato = dati["comments"].apply(preprocessa_testo).tolist()

corpus_tokenizzato = [c for c in corpus_tokenizzato if len(c) > 0]
print(f"Pulizia: elaborati {len(corpus_tokenizzato)} commenti validi.")

#COSTRUZIONE MODELLO WORD2VEC
# Inizializzaimo il Word2Vec
# https://radimrehurek.com/gensim/models/word2vec.html
modello = Word2Vec(vector_size=100,  # Dimensione degli embedding (100-200 per dataset piccoli/medi)
                   window=5,         # Contesto di 5 parole (ideale per commenti brevi)
                   min_count=5,      # Ignora parole con <5 occorrenze (riduce rumore)
                   sg=0,             # Modalità CBOW (più veloce e funziona bene con poco testo)
                   hs=0,             # Disabilita Hierarchical Softmax (usa Negative Sampling)
                   negative=10,      # Campiona 10 parole negative (bilancio qualità-velocità)
                   epochs=30,        # Più epoche per dataset piccoli (default=5)
                   workers=4,        # Threads per training parallelo
                   alpha=0.025,      # Learning rate iniziale
                   min_alpha=0.0001, # Learning rate finale (decresce linearmente)
                   cbow_mean=1       # Media dei vettori di contesto (default)
                   )

#Costruzione vocabolario 
modello.build_vocab(corpus_tokenizzato)

#addestramento 
modello.train(corpus_tokenizzato, total_examples=modello.corpus_count, epochs=30)

modello.save("bodycam_Gabby_model.model")
print("Modello salvato con successo!")

bodycam_model = Word2Vec.load("bodycam_Gabby_model.model")
parole = list(bodycam_model.wv.index_to_key)

#Cordinate vettori
vector = bodycam_model.wv["gabby"]
print(vector)

#Similarità del Coseno
word_vectors = bodycam_model.wv  # wv contiene i KeyedVectors
similarity = word_vectors.similarity('signs', 'calm')
print(similarity)


#ANALISI E CONFRONTO
print("\nESTRAZIONE CAMPI SEMANTICI")
vicini_gabby = modello.wv.most_similar(positive="gabby", topn=20)
vicini_brian = modello.wv.most_similar(positive="brian", topn=20)

df_confronto = pd.DataFrame({
    "Vicini a GABBY": [parola for parola, _ in vicini_gabby],
    "Sim. Gabby": [round(score, 3) for _, score in vicini_gabby],
    "Vicini a BRIAN": [parola for parola, _ in vicini_brian],
    "Sim. Brian": [round(score, 3) for _, score in vicini_brian]
})

print(df_confronto.head(10))

df_confronto.to_csv("campo_semantico_genere.csv", index=False)
print("\nFile 'campo_semantico_genere.csv' pronto")

# Analisi comparativa strutturata
dimensioni = {
    "Agency": ("control", "decision", "action", "calm"),
    "Patologia": ("unstable", "hysterical", "crazy", "emotional"),
    "Vittimizzazione": ("fear", "trapped", "help", "pain")
}

print(f"{'DIMENSIONE':<15} | {'GABBY (sim)':<12} | {'BRIAN (sim)':<12} | {'DELTA'}")
print("-" * 55)

for dim, parole in dimensioni.items():
    for p in parole:
        if p in bodycam_model.wv:
            s_g = bodycam_model.wv.similarity("gabby", p)
            s_b = bodycam_model.wv.similarity("brian", p)
            delta = s_g - s_b
            print(f"{p:<15} | {s_g:.3f}        | {s_b:.3f}        | {delta:+.3f}")

#Analisi termini chiave 
bodycam_model.wv.most_similar(positive = "police", topn = 15)

#Ritorno al corpus - analisi qualitativa 
# verifica empirica ipotesi "police" + "properly"
maschera_properly = dati['comments'].str.contains('police', case=False, na=False) & \
                    dati['comments'].str.contains('properly', case=False, na=False)

frasi_prova = dati[maschera_properly]['comments'].dropna().head(5).tolist()

print("\n=== PROVA EMPIRICA: COMMENTI REALI CON 'POLICE' E 'PROPERLY' ===")
for i, frase in enumerate(frasi_prova):
    print(f"\nCommento {i+1}:\n{frase}")  

# L'avverbio 'properly' co-occorre con il sostantivo 'police' prevalentemente 
# per denunciare una grave lacuna istituzionale: la mancanza di un addestramento 
# adeguato ('proper training') per la gestione e il riconoscimento delle 
# dinamiche di Domestic Violence (DV).

# Analisi "hysterical"
bodycam_model.wv.most_similar(positive = "hysterical", topn = 15)


# Distanza "dramatic" con Gabby e Brian 
print("\nESTRAZIONE CAMPI SEMANTICI")

sim_gabby_dramatic = bodycam_model.wv.similarity("gabby", "hysterical")
sim_brian_dramatic = bodycam_model.wv.similarity("brian", "hysterical")
    
print("\n=== TEST POLARIZZAZIONE: 'dramatic' ===")
print(f"Correlazione con GABBY: {sim_gabby_dramatic:.3f}")
print(f"Correlazione con BRIAN: {sim_brian_dramatic:.3f}")

# Distanza "blame" con Gabby e Brian 
print("\nESTRAZIONE CAMPI SEMANTICI")

# Salviamo le similarità nelle variabili con il nome corretto ("blame")
sim_gabby_blame = bodycam_model.wv.similarity("gabby", "blame")
sim_brian_blame = bodycam_model.wv.similarity("brian", "blame")
sim_police_blame = bodycam_model.wv.similarity("police", "blame")
  
print("\n=== TEST POLARIZZAZIONE: 'blame' ===")
print(f"Correlazione con GABBY: {sim_gabby_blame:.3f}")
print(f"Correlazione con BRIAN: {sim_brian_blame:.3f}")
print(f"Correlazione con POLICE: {sim_police_blame:.3f}")
# attribuzione colpa alla polizia 

# Doppio standard di genere 
print("\n=== TEST SUL DOPPIO STANDARD DI GENERE ===")

# Lista di termini per scovare il bias cognitivo e di genere
parole_bias = [
    "crazy", "dramatic", "poor", "emotional", "hysterical", 
    "calm", "polite", "charming", "joking",
    "manipulative", "believed", "instigator", "pure", "angel", "bautiful"
]

# Intestazione della tabella
print(f"{'PAROLA':<15} | {'SIM. GABBY':<10} | {'SIM. BRIAN':<10} | ESITO DEL BIAS")
print("-" * 65)

# Ciclo di analisi per ogni parola
for parola in parole_bias:
    # Controlliamo che la parola esista nel vocabolario del modello
    if parola in bodycam_model.wv.key_to_index:
        sim_g = bodycam_model.wv.similarity("gabby", parola)
        sim_b = bodycam_model.wv.similarity("brian", parola)
        
        # Determiniamo verso chi pende il pregiudizio
        if sim_g > sim_b:
            esito = "Pende verso GABBY"
        elif sim_b > sim_g:
            esito = "Pende verso BRIAN"
        else:
            esito = "Neutro"
            
        print(f"{parola:<15} | {sim_g:.3f}      | {sim_b:.3f}      | {esito}")
    else:
        print(f"{parola:<15} | --- Non abbastanza frequente nel dataset ---")


# VERIFICA EMPIRICA: La "calma" di Brian
# Cerchiamo i commenti grezzi che contengono "Brian" (o "he") e "calm" o "joking"

maschera_brian_calm = (dati['comments'].str.contains(r'\b(brian|he)\b', case=False, na=False, regex=True)) & \
                      (dati['comments'].str.contains(r'\b(calm|joking)\b', case=False, na=False, regex=True))

frasi_prova_bias = dati[maschera_brian_calm]['comments'].dropna().head(5).tolist()

print("\n=== PROVA EMPIRICA: LA 'CALMA' DI BRIAN NEI COMMENTI ===")
for i, frase in enumerate(frasi_prova_bias):
    print(f"\nCommento {i+1}:\n{frase}")
# percezione della calma come strumento manipolatorio 


# Associazione correlazioni parole Gabby e Brian
df_confronto = pd.DataFrame({
    "Vicini a GABBY": [parola for parola, _ in vicini_gabby],
    "Sim. Gabby": [round(score, 3) for _, score in vicini_gabby],
    "Vicini a BRIAN": [parola for parola, _ in vicini_brian],
    "Sim. Brian": [round(score, 3) for _, score in vicini_brian]
})

print(df_confronto.head(10))

# check per analizzare l'associazione tra calm + Brian
print("\n=== LA  NATURA DELLA 'CALMA' DI BRIAN ===")

# Uniamo i vettori di "brian" e "calm" per vedere cosa c'è in quello specifico spazio semantico
if "calm" in bodycam_model.wv.key_to_index and "brian" in bodycam_model.wv.key_to_index:
    vicini_brian_calm = bodycam_model.wv.most_similar(positive=["brian", "calm"], topn=10)
    
    print("Concetti associati alla combinazione [Brian + Calm]:\n")
    for parola, score in vicini_brian_calm:
        print(f"  - {parola:<15} ({score:.3f})")
else:
    print("Le parole non sono sufficientemente presenti nel vocabolario.")

# calma pubblica, performativa 

# check associazione hysterical + Gabby
print("\n=== LA  NATURA DELLA 'HYSTERICAL' DI GABBY ===")

# Uniamo i vettori di "gabby" e "hysterical" per vedere cosa c'è in quello specifico spazio semantico
if "hysterical" in bodycam_model.wv.key_to_index and "gabby" in bodycam_model.wv.key_to_index:
    vicini_brian_calm = bodycam_model.wv.most_similar(positive=["gabby", "hysterical"], topn=10)
    
    print("Concetti associati alla combinazione [gabby + hysterical]:\n")
    for parola, score in vicini_brian_calm:
        print(f"  - {parola:<15} ({score:.3f})")
else:
    print("Le parole non sono sufficientemente presenti nel vocabolario.") 

# validazione del trauma


''' GENERAZIONE GRAFICI'''

print("Generazione della Heatmap di similarità...")

from gensim.models import Word2Vec

model = Word2Vec.load("bodycam_Gabby_model.model")

# I 3 ATTORI (I soggetti del giudizio pubblico)

# 10 keyword
concetti = [
    "gabby", "brian", "police"
    # -- Blocco Colpa e Fallimento --
    "failed",         # Fallimento istituzionale
    "blame",          # Direzione della colpa
    "victim",         # Riconoscimento dello status
    "instigator",     # Riconoscimento della provocazione/gaslighting
    
    # -- Blocco Maschile: Dalla Performance alla Diagnosi --
    "calm",           # Il bias razionale
    "smiling",        # La performance teatrale
    "manipulative",   # La verità nascosta
    
    # -- Blocco Femminile: Dallo Stigma alla Validazione --
    "hysterical",     # Il bias patologizzante
    "distraught",     # Il dolore autentico
    "traumatized"     # La validazione clinica del pubblico
]

# Controllo vocabolario
parole = [p for p in parole if p in model.wv]

# Matrice similarità
matrice_similarita = np.zeros((len(parole), len(parole)))
for i, word1 in enumerate(parole):
    for j, word2 in enumerate(parole):
        matrice_similarita[i][j] = model.wv.similarity(word1, word2)
        if matrice_similarita[i][j] > 0.99:
            matrice_similarita[i][j] = 0

soglia = 0.11
mask_sotto_soglia = matrice_similarita <= soglia
annotations = np.where(
    matrice_similarita > soglia,
    np.round(matrice_similarita, 2),
    ""
)

plt.figure(figsize=(12, 10))
sns.set(font_scale=0.9)

ax = sns.heatmap(
    matrice_similarita,
    mask=mask_sotto_soglia,
    annot=annotations,
    fmt="",
    cmap="Reds",
    xticklabels=parole,
    yticklabels=parole,
    linewidths=1,
    cbar_kws={"label": "Word2Vec Similarity"}
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

plt.tight_layout()
plt.savefig("HEATmap.termini_def.png", dpi=300, bbox_inches="tight")

# No bias di genere come patologizzazione della vittima, colpa police, calma come strumento manipolatorio



# RIDURRE I DATI DI UN MODELLO W2V

# Importiamo le istruzioni per la riduzione dei dati: t-SNE
from sklearn.manifold import TSNE
# Importiamo una libreria per la gestione degli array
import numpy as np

# Creiamo una funzione che riduce le dimensioni del modello
def riduci_dimensioni(nuovo_modello):

    numero_dimensioni = 2

    vettori = []        # Lista vuota di vettori
    etichette = []      # Lista vuota di parole

    for parola in nuovo_modello.wv.index_to_key:
        vettori.append(nuovo_modello.wv[parola])
        etichette.append(parola)

    # Un array è una lista che può contenere solo un tipo di elementi
    # Gli array possono essere scritti in memoria in modo efficiente
    # .asarray trasforma le liste in array
    vettori = np.asarray(vettori)
    etichette = np.asarray(etichette)

    # t-distributed Stochastic Neighbor Embedding (t-SNE)
    # random_state corrisponde al seed (None = random)
    # .fit_transform genera le nuove coordinate dei fattori
    modello_tsne = TSNE(n_components=numero_dimensioni, random_state=2)
    vettori = modello_tsne.fit_transform(vettori)

    # Vettori e una lista di liste da due valori
    # Ritagliamo per x il 1° e per y il 2° valore
    x_vals = [v[0] for v in vettori]
    y_vals = [v[1] for v in vettori]

    return x_vals, y_vals, etichette

x_vals, y_vals, etichette = riduci_dimensioni(bodycam_model)


# GRAFICO WORD2VEC
def disegna_modello(x_vals, y_vals, etichette):
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from adjustText import adjust_text

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    random.seed(12345)
    global AVx
    global AVy
    AVx = []
    AVy = []

    # Inizializza il grafico (Ingrandito per maggiore respiro)
    plt.figure(figsize=(15,15))

    # Sceglie il tipo di grafico (Disegna solo i PUNTINI per tutte le parole)
    plt.scatter(x_vals, y_vals, s=10, alpha=0.3)

    # 1. CICLO PER IL SALVATAGGIO EXCEL (Senza stampare testo sul grafico!)
    for etica in range(len(etichette)):
        if abs(x_vals[etica]) > 2 or abs(y_vals[etica]) > 2:
            # Abbiamo rimosso plt.annotate da qui!

            duox = [x_vals[etica], etichette[etica]]
            duoy = [y_vals[etica], etichette[etica]]

            AVx.append(duox)
            AVy.append(duoy)

    # parole chiave
    parole_chiave = [
        "police", "officer", "cop", "law", "authority",
        "failed", "ignored", "accountability", "justice", "negligence",
        "gabby","fiancée", "daughter", "woman", "girl",
        "brian", "man", "boyfriend",
        "hysterical", "emotional", "crazy", "anxiety", "unstable",
        "killer", "murderer", "guilty", "violent", "abuser",
        "calm", "rational", "controlled", "composed", "logical",
        "blame", "fault", "responsible", "victim", "aggressor",
        "abuse", "domestic", "coercive", "gaslight", "toxic",
        "believe", "truth", "lie", "credible", "manipulative", 
        "credit", "cards", "bank", "separates", "interviewing", "scene", "pushed", 
        "blog", "worked", "mother", "father",           
        "deleted", "redact", "outcry", "heartache",    
        "puppet", "fraudulent",    
        "nuance",                    
        "scene", "separates"
    ]
    parole_chiave_sub = parole_chiave 


    testi_da_stampare = []
    # 3. STAMPA DELLE ETICHETTE (Senza offset manuali!)
    for i in range(len(parole_chiave)):
        indiceextra = np.where(etichette==parole_chiave[i])

        if indiceextra[0].size > 0:
            indiceextra = indiceextra[0][0]

            # Creiamo l'etichetta base e la infiliamo nel nostro contenitore
            testo = plt.text(x_vals[indiceextra], y_vals[indiceextra],
                             parole_chiave_sub[i],
                             color="#37474F", fontsize=14, fontweight='bold',
                             bbox=dict(facecolor='#FDF5E6', alpha=0.8, edgecolor='none', pad=1))
            testi_da_stampare.append(testo)

    print("Calcolo delle posizioni anti-sovrapposizione in corso...")
    adjust_text(testi_da_stampare,
                arrowprops=dict(arrowstyle="->", color='#E64A19', lw=0.5), # Disegna una lineetta se sposta molto la parola
                expand_points=(1.2, 1.2), # Margine di respiro tra le parole
                expand_text=(1.2, 1.2))

    # Disegniamo gli assi
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.axvline(0, color='gray', linewidth=1, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=15)

    # Salvataggio e visualizzazione
    plt.savefig("Grafico_WORD2VEC.png", dpi=300, bbox_inches='tight')
    plt.show()

# Esecuzione
disegna_modello(x_vals, y_vals, etichette)
   
 # 2. Mostra il grafico dentro Spyder
plt.show()

# Salvataggio Excel (mantenuto intatto fuori dalla funzione)

Asse_X = pd.DataFrame(sorted(AVx))
Asse_X.to_excel("AsseX.xlsx")
Asse_Y = pd.DataFrame(sorted(AVy))
Asse_Y.to_excel("AsseY.xlsx")

#Annalisi quadrante in basso a sinistra 
print("\n=== SCANNER RADAR: QUADRANTE IN BASSO A SINISTRA (X < 0, Y < 0) ===")

parole_basso_sinistra = []

for i in range(len(etichette)):
    # Filtra rigorosamente il quadrante inferiore sinistro
    if x_vals[i] < 0 and y_vals[i] < 0:
        # Calcoliamo la distanza dall'origine del quadrante per trovare le parole più centrali
        distanza = np.sqrt(x_vals[i]**2 + y_vals[i]**2)
        parole_basso_sinistra.append((etichette[i], x_vals[i], y_vals[i], distanza))

# Ordina le parole in base alla loro centralità nel quadrante
parole_basso_sinistra.sort(key=lambda x: x[3], reverse=True)

print(f"Trovate {len(parole_basso_sinistra)} parole globali in questo quadrante.")
print("\nEcco le prime 30 parole che popolano la nuvola blu in basso a sinistra:\n")

for p in parole_basso_sinistra[:30]:
    print(f"  - {p[0]:<20} [X: {p[1]:.2f}, Y: {p[2]:.2f}]")


#Annalisi nucleo centrale
print("\n=== SCANNER RADAR: IL NUCLEO CENTRALE (X=0, Y=0) ===")

parole_centro = []

for i in range(len(etichette)):
    # Calcoliamo la distanza dal centro esatto (teorema di Pitagora)
    distanza_dal_centro = np.sqrt(x_vals[i]**2 + y_vals[i]**2)
    
    # Filtriamo solo le parole che vivono in un raggio strettissimo (es. meno di 3 unità dal centro)
    if distanza_dal_centro < 3.0: 
        parole_centro.append((etichette[i], x_vals[i], y_vals[i], distanza_dal_centro))

# Ordiniamo le parole partendo da quella matematicamente più vicina allo zero
parole_centro.sort(key=lambda x: x[3])

print(f"Trovate {len(parole_centro)} parole nel raggio centrale.")
print("\nEcco le prime 30 parole che compongono il 'buco nero' al centro del grafico:\n")

for p in parole_centro[:30]:
    print(f"  - {p[0]:<20} [X: {p[1]:.2f}, Y: {p[2]:.2f}, Distanza: {p[3]:.2f}]")



#aggiungi grafico e sistema heatmap 