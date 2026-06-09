# ==========================================
# 1. CREAZIONE E PULIZIA DEI DATI (Data Cleaning)
# ==========================================
library(quanteda)

dati = read.csv("dataset_comments_bodycam.csv")
# Creazione del corpus
testi <- corpus(dati, text_field="comments")

# Tokenizzazione Avanzata
# Aggiunta la rimozione di numeri, simboli e link web
testi.tokenizzati <- tokens(testi, 
                            remove_punct = TRUE, 
                            remove_numbers = TRUE, 
                            remove_symbols = TRUE,
                            remove_url = TRUE)

# Rimuovi tutte le parole che iniziano con "@"
testi.tokenizzati <- tokens_remove(testi.tokenizzati, pattern = "@*")

# Uniformiamo tutto il testo in minuscolo
testi.tokenizzati <- tokens_tolower(testi.tokenizzati)

# Stemming (riduce le parole alla radice, es. "working" -> "work")
# Step fondamentale per migliorare la qualità dei topic nell'LDA
testi.puliti <- tokens_wordstem(testi.tokenizzati, language = "en")

# Rimozione Stopwords e parole custom (ottimizzato in un solo passaggio)
parole_da_rimuovere <- c(stopwords("en"), 
                         "like", 
                         "just", 
                         "also", 
                         "much", 
                         "it", 
                         "many",
                         "one", 
                         "don", 
                         "even", 
                         "something"
                         )
testi.puliti <- tokens_remove(testi.tokenizzati, parole_da_rimuovere)

# Crea matrice bag of words (documenti x termini)
matrice <- dfm(testi.puliti)

# Pruning Avanzato
# 1. Rimuove le "paroline" corte (meno di 3 caratteri)
matrice <- dfm_keep(matrice, min_nchar = 3) 
# 2. Pruning per frequenza (come avevi fatto tu)
matrice <- dfm_trim(matrice, min_termfreq=5, min_docfreq=2)


# ==========================================
# 2. PREPARAZIONE E ADDESTRAMENTO LDA
# ==========================================
library(topicmodels)

# CORREZIONE CRITICA: Rimuoviamo i documenti vuoti (0 token) PRIMA dell'LDA.
# Dopo aver tolto punteggiatura, stopwords e fatto il pruning, alcuni commenti 
# potrebbero essere diventati completamente vuoti. Vanno eliminati dalla matrice.
matrice <- dfm_subset(matrice, ntoken(matrice) > 0)

# PREPARA LA MATRICE PER LA LDA
matrice.lda <- convert(matrice, to="topicmodels")

# EFFETTUA LA LDA
# method = "Gibbs", k = 5, alpha = 0.1
m <- LDA(matrice.lda, method="Gibbs", k=5, control=list(alpha=0.1, seed = 42))

# Mostra i primi 5 termini di ogni topic 
terms(m, 5)


# ==========================================
# 3. VISUALIZZAZIONE CON LDAvis
# ==========================================
library(LDAvis)

# Non serve più rimuovere le righe con slam::row_sums perché lo abbiamo fatto 
# a monte sul DFM tramite `dfm_subset()`. Questo rende il codice più pulito.
phi <- as.matrix(posterior(m)$terms)
theta <- as.matrix(posterior(m)$topics)
vocab <- colnames(phi)

# Estrazione delle frequenze e lunghezze
doc.length <- slam::row_sums(matrice.lda)
term.freq <- slam::col_sums(matrice.lda)[match(vocab, colnames(matrice.lda))]

# Creazione JSON e Visualizzazione
json <- createJSON(phi = phi, 
                   theta = theta, 
                   vocab = vocab,
                   doc.length = doc.length, 
                   term.frequency = term.freq)
serVis(json)

