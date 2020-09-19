
##############################################################################
# IMPORTAR BIBLIOTECAS
# para manipular dados
import pandas as pd

# para expressão regular
import re

# para remover pontuações
import string

# para manipular emoji/emoticon
from emot.emo_unicode import UNICODE_EMO

# para PLN
import spacy
import pt_core_news_md

# para carregar os artefatos
import pickle

# esta aplicação
import streamlit as st

##############################################################################

# dicionários com as interpretações, um para emojis outro para emoticons
dict_emojis = {
    'exclamation_question_mark': 'ruim',
    'person_pouting': 'ruim',
    'kiss_mark': 'ótimo',
    'upside-down_face': 'ótimo',
    'smiling_face_with_open_mouth_&_smiling_eyes': 'ótimo',
    'love_letter': 'ótimo',
    'rose': 'ótimo',
    'angry_face_with_horns': 'ruim',
    'yellow_heart': 'ótimo',
    'blue_heart': 'ótimo',
    'green_heart': 'ótimo',
    'relieved_face': 'ótimo',
    'trophy': 'ótimo',
    'expressionless_face': 'ruim',
    'slightly_smiling_face': 'ótimo',
    'nauseated_face': 'ruim',
    'face_with_stuck-out_tongue_&_winking_eye': 'ótimo',
    'OK_hand': 'ótimo',
    'neutral_face': 'ruim',
    'person_shrugging': 'ruim',
    'weary_face': 'ruim',
    'heart_with_arrow': 'ótimo',
    'grimacing_face': 'ruim',
    'sleepy_face': 'ruim',
    'pig_face': 'ruim',
    'thinking_face': 'ruim',
    'loudly_crying_face': 'ruim',
    'blossom': 'ótimo',
    'face_with_cold_sweat': 'ruim',
    'crying_cat_face': 'ruim',
    'unamused_face': 'ruim',
    'disappointed_but_relieved_face': 'ruim',
    'smiling_face': 'ótimo',
    'face_screaming_in_fear': 'ruim',
    'face_with_steam_from_nose': 'ruim',
    'broken_heart': 'ruim',
    'see-no-evil_monkey': 'ruim',
    'two_hearts': 'ótimo',
    'growing_heart': 'ótimo',
    'slightly_frowning_face': 'ruim',
    'crying_face': 'ruim',
    'dizzy': 'ruim',
    'smiling_face_with_open_mouth_&_closed_eyes': 'ótimo',
    'victory_hand': 'ótimo',
    'face_with_rolling_eyes': 'ruim',
    'revolving_hearts': 'ótimo',
    'smiling_face_with_open_mouth': 'ótimo',
    'rolling_on_the_floor_laughing': 'ótimo',
    'pensive_face': 'ruim',
    'dizzy_face': 'ruim',
    'angry_face': 'ruim',
    'confused_face': 'ruim',
    'smiling_face_with_open_mouth_&_cold_sweat': 'ótimo',
    'smirking_face': 'ótimo',
    'smiling_face_with_sunglasses': 'ótimo',
    'face_with_tears_of_joy': 'ótimo',
    'white_medium_star': 'ótimo',
    'thumbs_down': 'ruim',
    'red_heart': 'ótimo',
    'clapping_hands': 'ótimo',
    'smiling_face_with_halo': 'ótimo',
    'purple_heart': 'ótimo',
    'smiling_face_with_heart-eyes': 'ótimo',
    'heart_suit': 'ótimo',
    'hugging_face': 'ótimo',
    'glowing_star': 'ótimo',
    'smiling_face_with_smiling_eyes': 'ótimo',
    'grinning_face_with_smiling_eyes': 'ótimo',
    'thumbs_up': 'ótimo',
    'face_blowing_a_kiss': 'ótimo',
    'winking_face': 'ótimo'
}

dict_emotis = {
    'Wink or smirk': 'ótimo',
    'Happy face or smiley': 'ótimo',
    'Tongue sticking out, cheeky, playful or blowing a raspberry': 'ótimo',
    'Frown, sad, andry or pouting': 'ruim',
    'Skeptical, annoyed, undecided, uneasy or hesitant': 'ruim'
}


# lista de pontuações    
pontuacoes = string.punctuation
    
# lista com stop words
ls_stop_words = pickle.load(open('Artefatos/ls_stop_words.pkl', 'rb'))

# vetorizadores
vect_criticas = pickle.load(open('Artefatos/vect_criticas.pkl', 'rb'))
vect_estrelas = pickle.load(open('Artefatos/vect_estrelas.pkl', 'rb'))
    
# normalizadores
scaler_criticas = pickle.load(open('Artefatos/scaler_criticas.pkl', 'rb'))
scaler_estrelas = pickle.load(open('Artefatos/scaler_estrelas.pkl', 'rb'))
    
# modelos de Machine Learning
modelo_criticas = pickle.load(open('Artefatos/grad_boost_class.pkl', 'rb'))
modelo_estrelas = pickle.load(open('Artefatos/naive_bayes.pkl', 'rb'))

# modelo português PLN do Spacy
# cria o objeto de pré processamento spacy
pln = pt_core_news_md.load()
    
    
# transformar emoji/emoticon para seu significado literal
def traduzir_emoti_emoji(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, UNICODE_EMO[emot])
        text = text.replace(':', ' ')
    lista = text.split(' ')
    for x in range(len(lista)):
        chave = lista[x]
        if chave in dict_emojis:
            lista[x] = dict_emojis[chave]
        if chave in dict_emotis:
            lista[x] = dict_emotis[chave]
        
    texto = ' '
    texto = (texto.join(lista)) 
    texto = texto.strip()    
    return texto
    

# função final de préprocessamento
def preprocessamento(texto):
    texto = str(texto).lower()

    texto = traduzir_emoti_emoji(texto)
    texto = re.sub(r" +", ' ', texto)
    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)
    lista = [palavra for palavra in lista if palavra not in ls_stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()]).strip()
    return lista
    
    
# funções de feature engineering
def qtde_palavras(texto):
    texto = str(texto)
    palavras = texto.split()
    return len(palavras)

def qtde_maiusculas(texto):
    texto = str(texto)
    quantidade = 0
    for c in texto:
        if c.isupper():
            quantidade += 1
    return quantidade

def comprimento(texto):
    texto = str(texto)
    return len(texto)

def qtde_exclamacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '!' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_interrogacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '?' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_pontuacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '.' in texto[x] or ',' in texto[x] or ';' in texto[x] or ':' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_simbolos(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '*' in texto[x] or '&' in texto[x] or '%' in texto[x] or '$' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_palavras_unicas(texto):
    texto = str(texto)
    palavras = texto.split()
    palavras_unicas = set(palavras)
    return len(palavras_unicas)

def qtde_tag_part_of_speech(texto):
    texto = str(texto)
    doc = pln(texto)
    pos_list = [token.pos_ for token in doc]
    qtde_substantivos = len([w for w in pos_list if w == 'NOUN'])
    qtde_adjetivos    = len([w for w in pos_list if w == 'ADJ'])
    qtde_verbos       = len([w for w in pos_list if w == 'VERB'])
    qtde_adverbios    = len([w for w in pos_list if w == 'ADV'])
    qtde_interjeicoes = len([w for w in pos_list if w == 'INTJ'])
    return[qtde_substantivos, qtde_adjetivos, qtde_verbos, qtde_adverbios, qtde_interjeicoes]
    
    
# função para previsoes
def predict_probas(texto):
    d = {'Comentario': [texto]}
    resenha_df = pd.DataFrame(data=d)
    
    # aplicar a função de préprocessamento
    resenha_df['coment_processado'] = resenha_df['Comentario'].apply(preprocessamento)
           
    # criar novas variáeis
    resenha_df['qtde_palavras']    = resenha_df['Comentario'].apply(qtde_palavras)
    resenha_df['qtde_maiusculas']  = resenha_df['Comentario'].apply(qtde_maiusculas)
    resenha_df['comprimento']      = resenha_df['Comentario'].apply(comprimento)
    resenha_df['maiusc_x_compri']  = resenha_df['qtde_maiusculas'] / resenha_df['comprimento']
    resenha_df['qtde_exclamacoes'] = resenha_df['Comentario'].apply(qtde_exclamacoes)
    resenha_df['qtde_interrogacoes'] = resenha_df['Comentario'].apply(qtde_interrogacoes)
    resenha_df['qtde_pontuacoes']    = resenha_df['Comentario'].apply(qtde_pontuacoes)
    resenha_df['qtde_simbolos']      = resenha_df['Comentario'].apply(qtde_simbolos)
    resenha_df['qtde_palavras_unicas'] = resenha_df['Comentario'].apply(qtde_palavras_unicas)
    resenha_df['unicas_x_comprimento'] = resenha_df['qtde_palavras_unicas'] / resenha_df['comprimento']
    resenha_df['qtde_substantivos'], resenha_df['qtde_adjetivos'], resenha_df['qtde_verbos'], resenha_df['qtde_adverbios'], resenha_df['qtde_interjeicoes'] = zip(*resenha_df['Comentario'].apply(lambda comment: qtde_tag_part_of_speech(comment)))
    resenha_df['substantivos_vs_comprimento']  = resenha_df['qtde_substantivos'] / resenha_df['comprimento']
    resenha_df['adjectivos_x_comprimento']     = resenha_df['qtde_adjetivos'] / resenha_df['comprimento']
    resenha_df['verbos_x_comprimento']         = resenha_df['qtde_verbos'] /resenha_df['comprimento']
    resenha_df['adverbios_x_comprimento']      = resenha_df['qtde_adverbios'] /resenha_df['comprimento']
    resenha_df['interjeicoes_x_comprimento']   = resenha_df['qtde_interjeicoes'] /resenha_df['comprimento']
    resenha_df['substantivos_x_qtde_palavras'] = resenha_df['qtde_substantivos'] / resenha_df['qtde_palavras']
    resenha_df['adjectivos_x_qtde_palavras']   = resenha_df['qtde_adjetivos'] / resenha_df['qtde_palavras']
    resenha_df['verbos_x_qtde_palavras']       = resenha_df['qtde_verbos'] / resenha_df['qtde_palavras']
    resenha_df['adverbios_x_qtde_palavras']    = resenha_df['qtde_adverbios'] / resenha_df['qtde_palavras']
    resenha_df['interjeicoes_x_qtde_palavras'] = resenha_df['qtde_interjeicoes'] / resenha_df['qtde_palavras']        
    
    # vetorizar os dados para classificar criticas
    X = vect_criticas.transform(resenha_df.coment_processado)
    # Gerar Pandas DataFrame
    resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect_criticas.get_feature_names())
    
    # anexar os dados
    resenhas_tfidf['qtde_palavras']                = resenha_df['qtde_palavras']                 
    resenhas_tfidf['qtde_maiusculas']              = resenha_df['qtde_maiusculas']               
    resenhas_tfidf['comprimento']                  = resenha_df['comprimento']                   
    resenhas_tfidf['maiusc_x_compri']              = resenha_df['maiusc_x_compri']               
    resenhas_tfidf['qtde_exclamacoes']             = resenha_df['qtde_exclamacoes']              
    resenhas_tfidf['qtde_interrogacoes']           = resenha_df['qtde_interrogacoes']            
    resenhas_tfidf['qtde_pontuacoes']              = resenha_df['qtde_pontuacoes']               
    resenhas_tfidf['qtde_simbolos']                = resenha_df['qtde_simbolos']                 
    resenhas_tfidf['qtde_palavras_unicas']         = resenha_df['qtde_palavras_unicas']          
    resenhas_tfidf['unicas_x_comprimento']         = resenha_df['unicas_x_comprimento']          
    resenhas_tfidf['qtde_substantivos']            = resenha_df['qtde_substantivos']     
    resenhas_tfidf['qtde_adjetivos']               = resenha_df['qtde_adjetivos']      
    resenhas_tfidf['qtde_verbos']                  = resenha_df['qtde_verbos']        
    resenhas_tfidf['qtde_adverbios']               = resenha_df['qtde_adverbios']              
    resenhas_tfidf['qtde_interjeicoes']            = resenha_df['qtde_interjeicoes']              
    resenhas_tfidf['substantivos_vs_comprimento']  = resenha_df['substantivos_vs_comprimento']   
    resenhas_tfidf['adjectivos_x_comprimento']     = resenha_df['adjectivos_x_comprimento']      
    resenhas_tfidf['verbos_x_comprimento']         = resenha_df['verbos_x_comprimento']          
    resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['adverbios_x_comprimento']
    resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['interjeicoes_x_comprimento']
    resenhas_tfidf['substantivos_x_qtde_palavras'] = resenha_df['substantivos_x_qtde_palavras']  
    resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenha_df['adjectivos_x_qtde_palavras']    
    resenhas_tfidf['verbos_x_qtde_palavras']       = resenha_df['verbos_x_qtde_palavras']       
    resenhas_tfidf['adverbios_x_qtde_palavras']    = resenha_df['adverbios_x_qtde_palavras']
    resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenha_df['interjeicoes_x_qtde_palavras']        
    
    # normalizar os dados para classificar criticas
    X = scaler_criticas.transform(resenhas_tfidf.fillna(0))
    
    # probabilidade de ser crítica
    proba_critica = round(modelo_criticas.predict_proba(X)[0][1], 4) * 100
    
    # vetorizar os dados para classificar estrelas
    X = vect_estrelas.transform(resenha_df.coment_processado)
    
    # Gerar Pandas DataFrame
    resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect_estrelas.get_feature_names())
    
    # anexar os dados
    resenhas_tfidf['qtde_palavras']                = resenha_df['qtde_palavras']                 
    resenhas_tfidf['qtde_maiusculas']              = resenha_df['qtde_maiusculas']               
    resenhas_tfidf['comprimento']                  = resenha_df['comprimento']                   
    resenhas_tfidf['maiusc_x_compri']              = resenha_df['maiusc_x_compri']               
    resenhas_tfidf['qtde_exclamacoes']             = resenha_df['qtde_exclamacoes']              
    resenhas_tfidf['qtde_interrogacoes']           = resenha_df['qtde_interrogacoes']            
    resenhas_tfidf['qtde_pontuacoes']              = resenha_df['qtde_pontuacoes']               
    resenhas_tfidf['qtde_simbolos']                = resenha_df['qtde_simbolos']                 
    resenhas_tfidf['qtde_palavras_unicas']         = resenha_df['qtde_palavras_unicas']          
    resenhas_tfidf['unicas_x_comprimento']         = resenha_df['unicas_x_comprimento']          
    resenhas_tfidf['qtde_substantivos']            = resenha_df['qtde_substantivos']     
    resenhas_tfidf['qtde_adjetivos']               = resenha_df['qtde_adjetivos']      
    resenhas_tfidf['qtde_verbos']                  = resenha_df['qtde_verbos']        
    resenhas_tfidf['qtde_adverbios']               = resenha_df['qtde_adverbios']              
    resenhas_tfidf['qtde_interjeicoes']            = resenha_df['qtde_interjeicoes']              
    resenhas_tfidf['substantivos_vs_comprimento']  = resenha_df['substantivos_vs_comprimento']   
    resenhas_tfidf['adjectivos_x_comprimento']     = resenha_df['adjectivos_x_comprimento']      
    resenhas_tfidf['verbos_x_comprimento']         = resenha_df['verbos_x_comprimento']          
    resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['adverbios_x_comprimento']
    resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['interjeicoes_x_comprimento']
    resenhas_tfidf['substantivos_x_qtde_palavras'] = resenha_df['substantivos_x_qtde_palavras']  
    resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenha_df['adjectivos_x_qtde_palavras']    
    resenhas_tfidf['verbos_x_qtde_palavras']       = resenha_df['verbos_x_qtde_palavras']       
    resenhas_tfidf['adverbios_x_qtde_palavras']    = resenha_df['adverbios_x_qtde_palavras']
    resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenha_df['interjeicoes_x_qtde_palavras']                
    
    # normalizar os dados para classificar estrelas
    X = scaler_estrelas.transform(resenhas_tfidf.fillna(0))
    
    # probabilidade de ser 5 estrelas
    proba_5_estrelas = round(modelo_estrelas.predict_proba(X)[0][1], 4) * 100
    
    return proba_critica, proba_5_estrelas
        
    
##############################################################################
# APLICACAO STREAMLIT
# Título
st.title('Classificador de Resenha de Apps')
    
# Receber a resenha
st.write('Digite uma resenha para obter as probabilidades')
resenha = st.text_input('')

if st.button('Calcular probabilidades', 'button'):
    tamanho_resenha = len(resenha)
    
    if tamanho_resenha < 2:
        st.error('Informe resenha acima para gerar probabilidades')
    elif tamanho_resenha >= 2:
        proba_critica, proba_5_estrelas = predict_probas(resenha)
    
        proba_critica = round(proba_critica, 1)
        proba_5_estrelas = round(proba_5_estrelas, 1)

        st.write('Probabilidade da resenha ser crítica: {}%'.format(proba_critica))
        st.write('Probabilidade de atribuir 5 estrelas: {}%'.format(proba_5_estrelas))