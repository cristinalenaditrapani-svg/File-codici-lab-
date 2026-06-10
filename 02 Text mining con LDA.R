# ==========================================
# 1. CREAZIONE E PULIZIA DEI DATI (Data Cleaning)
# ==========================================
library(quanteda)

# Caricamento dataset
dati = read.csv("dataset_comments_bodycam.csv")

# Creazione del corpus
testi <- corpus(dati, text_field="comments")

# Tokenizzazione Avanzata (rimozione punteggiatura, numeri, simboli e URL)
# Nota: remove_punct rimuove anche gli apostrofi separando o unendo le parole
testi.tokenizzati <- tokens(testi, 
                            remove_punct = TRUE, 
                            remove_numbers = TRUE, 
                            remove_symbols = TRUE,
                            remove_url = TRUE)

# Rimuovi tutte le parole che iniziano con "@"
testi.tokenizzati <- tokens_remove(testi.tokenizzati, pattern = "@*")

# Uniformiamo tutto il testo in minuscolo
testi.tokenizzati <- tokens_tolower(testi.tokenizzati)

# Rimozione Stopwords e parole custom
# NOTA: Le parole con apostrofo vanno inserite SENZA apostrofo (es. "dont", "cant") 
# perché remove_punct = TRUE elimina la punteggiatura prima di questo controllo.
parole_da_rimuovere <- c(stopwords("en"), 
                         "like", "just", "also", "much", "it", "many",
                         "one", "don", "even", "something",
                         "dont", "cant", "im", "youre", "didnt" # Esempi custom senza apostrofo
)

# PASSO 1: Rimuoviamo le stopwords e i termini custom dal testo minuscolo
testi.senza_stop <- tokens_remove(testi.tokenizzati, parole_da_rimuovere)

# PASSO 2: Applichiamo lo stemming (corretto l'input in 'testi.senza_stop')
# Questo ridurrà "abuse", "abusive", "abuser" alla radice unica "abus"
testi.puliti <- tokens_wordstem(testi.senza_stop, language = "en")

# Crea matrice bag of words (documenti x termini)
matrice <- dfm(testi.puliti)

# Pruning Avanzato
# 1. Rimuove le "paroline" corte (meno di 3 caratteri)
matrice <- dfm_keep(matrice, min_nchar = 3) 
# 2. Pruning per frequenza
matrice <- dfm_trim(matrice, min_termfreq = 5, min_docfreq = 2)


# ==========================================
# 2. PREPARAZIONE E ADDESTRAMENTO LDA
# ==========================================
library(topicmodels)

# Riconoscimento e rimozione dei documenti rimasti vuoti dopo la pulizia
matrice <- dfm_subset(matrice, ntoken(matrice) > 0)

# Prepara la matrice per la LDA
matrice.lda <- convert(matrice, to = "topicmodels")

# Effettua la LDA (Gibbs sampling, 5 topic)
m <- LDA(matrice.lda, method = "Gibbs", k = 5, control = list(alpha = 0.1, seed = 42))

# Mostra i primi 5 termini di ogni topic nella console
terms(m, 5)


# ==========================================
# 3. VISUALIZZAZIONE CON LDAvis
# ==========================================
library(LDAvis)

# Estrazione delle matrici dei parametri dal modello stimato
phi <- as.matrix(posterior(m)$terms)
theta <- as.matrix(posterior(m)$topics)

# CORREZIONE: Definiamo prima il vocabolario per evitare errori di sequenza
vocab <- colnames(phi)

# Estrazione corretta delle lunghezze dei documenti e frequenze totali dei termini
doc.length <- slam::row_sums(matrice.lda)
term.freq <- slam::col_sums(matrice.lda)[match(vocab, colnames(matrice.lda))]

# Creazione dell'oggetto JSON per l'interfaccia web
json <- createJSON(phi = phi, 
                   theta = theta, 
                   vocab = vocab,
                   doc.length = doc.length, 
                   term.frequency = term.freq)

# Apertura del grafico interattivo nel browser o nel pannello Viewer
serVis(json)
