#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:32:24 2020

@author: angelojr
"""

##############################################################################
# IMPORT PACKAGES
import joblib
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
import streamlit         as st

##############################################################################
# FUNÇÕES
def main():
    
# AED dos clientes
    def aed_data(data):
        st.write('Estado de atuação')
        plt.figure(figsize=(30,15))
        ax = sns.countplot(data=data, x='sg_uf')
        st.pyplot()
    
        st.write('Saúde tributária')
        plt.figure(figsize=(30,15))
        ax = sns.countplot(data=data, x='de_saude_tributaria')
        st.pyplot()
    
        st.write('Possuem e-mail?')
        plt.figure(figsize=(30,5))
        sns.countplot(data = data, y = 'fl_email').set_title('fl_email')
        st.pyplot()
    
        st.write('Possuem telefone?')
        plt.figure(figsize=(30,5))
        sns.countplot(data = data, y = 'fl_telefone').set_title('fl_telefone')
        st.pyplot()
            
        st.write('Idade')
        plt.figure(figsize=(30,5))
        sns.distplot(data.idade_empresa_anos, bins=20, kde=False, rug=True)
        st.pyplot()
        
        st.write('Setor de atuação')
        plt.figure(figsize=(30,5))
        sns.countplot(data = data, y = 'setor').set_title('setor')
        st.pyplot()
    
        st.write('Segmento de atuação')
        plt.figure(figsize=(30,20))
        sns.countplot(data = data, y = 'nm_segmento').set_title('segmento')
        st.pyplot()
            
        st.write('Faturamento estimado do grupo empresarial')
        plt.figure(figsize=(30,5))
        sns.distplot(data.vl_faturamento_estimado_grupo_aux, bins=20, kde=False, 
                     rug=True)
        st.pyplot()
        
        
    # selecionar clientes
    def selecionar_clientes(cluster, dist_para_centroide):
        # Todo o conjunto de dados do cluster do id selecionado
        dist_centroide = data[data['cluster'] == 
                                 cluster]['dist_centro']
        # selecionando íncides dos 25 mais próximos com distâncias menores:
        indices_selecionados_menores = \
        dist_centroide[dist_centroide < 
                           dist_para_centroide].sort_values(ascending=False).head(25).index
        # selecionando íncides dos 25 mais próximos com distâncias maiores:
        indices_selecionados_maiores = \
        dist_centroide[dist_centroide > 
                           dist_para_centroide].sort_values().head(25).index
        # criar dataframes com as seleções
        df1 = data.iloc[indices_selecionados_menores]
        df2 = data.iloc[indices_selecionados_maiores]
        # e juntar estes dataframes em 1
        df3 = pd.concat([df1, df2], ignore_index=True)
            
        st.success('Clientes localizados')
            
        # perfil dos clientes selecionados
        st.subheader('Perfil dos clientes selecionados')
        aed_data(df3)
           
        # fazer dowload de arquivos
        def get_binary_file_downloader_html(bin_file, file_label='File'):
            import os
            import base64
            ls_nao_salvar = ['cluster', 'dist_centro']
            df3.drop(ls_nao_salvar, axis=1, inplace=True)
            df3.to_csv('clientes_selecionados.csv', index=False)
            with open(bin_file, 'rb') as f:
                data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Baixar {file_label}</a>'
                return href
        
        # Fazer download da lista de recomendações
        st.write('Fazer download da lista de clientes')
        st.markdown(get_binary_file_downloader_html('clientes_selecionados.csv', 'arquivo'), unsafe_allow_html=True)
    ##############################################################################
    # IMPORT DATASET
    data = pd.read_csv('data/mercado.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    
    ##############################################################################
    # Título
    st.title('Sistema de Recomendação de Leads')
    
    # Advertência
    st.write('Atenção: este sistema foi criado como propósito de projeto final \
             do programa AceleraDev Data Science da Code Nation, sendo este seu \
             único fim. Se julgar o aqui apresentado relevante, acesse o \
             GitHub da aplicação')
             
    # Amostra dos dados
    st.subheader('Amostra dos dados')
    ls_nao_exibir = ['id', 'cluster', 'dist_centro']
    st.write(data.drop(ls_nao_exibir, axis=1).head(10))
    
    # perfil dos clientes
    st.subheader('Perfil dos clientes')
    aed_data(data)
    
    # Obter recomendações
    st.subheader('Obter recomendações')
    st.write('Os clientes estão dividos em clusters (agrupamentos). \
    Para obter a lista de recomendações, primeiro escolha \
    entre selecionar um cliente ou informar as características de um cliente \
    (esta segunda opção exigirá o preenchimento de vários campos)')
    
    opcao = st.radio("Como deseja obter recomendações?",
        ('Informar id de cliente', 'Informar características de um cliente'))
    
    if opcao == 'Informar id de cliente':
        id = st.text_input('Informe o id do cliente')
        id_localizado = False
        
        try:
            cluster_do_id = data[data['id'] == id]['cluster'].values[0]
            id_localizado = True
        except:
            st.error('Cliente não encontrado')
            
        if id_localizado:
             # Captura a distância do id para o centroide de seu cluster
            dist_centroide_id = data[data['id'] == id]['dist_centro'].values[0]
            # Todo o conjunto de dados do cluster do id selecionado
            distancias_cluster = data[data['cluster'] == 
                                     cluster_do_id]['dist_centro']
            
            selecionar_clientes(cluster_do_id, dist_centroide_id)
    else:
        st.write('Informe os dados abaixo')
        
        # UF
        sg_uf = st.selectbox('UF', ['AC', 'AM', 'MA', 'PI', 'RN', 'RO'])
        
        # Idade da empresa
        idade_empresa_anos = st.number_input('Idade da empresa (em anos)')
        if idade_empresa_anos < 0.0164:
            idade_empresa_anos = 0.0164
        if idade_empresa_anos > 106432:
            idade_empresa_anos = 106432
        
        # e-mail
        fl_email = st.radio("Possui e-mail?", ('Sim', 'Não'))
        if fl_email == 'Sim':
            fl_email = 1
        else:
            fl_email = 0
            
        # telefone
        fl_telefone = st.radio("Possui telefone?", ('Sim', 'Não'))
        if fl_telefone == 'Sim':
            fl_telefone = 1
        else:
            fl_telefone = 0
            
        # fl_rm
        fl_rm = st.radio("Localizada em região metropolitana?", ('Sim', 'Não'))
        if fl_rm == 'Sim':
            fl_rm = 1
        else:
            fl_rm = 0
            
        # Possui veículo
        fl_veiculo = st.radio("Possui veículo?", ('Sim', 'Não'))
        if fl_veiculo == 'Sim':
            fl_veiculo = 1
            # quantos são pesados
            vl_total_veiculos_pesados_grupo = st.number_input('Valor dos veículos pesados')
            if vl_total_veiculos_pesados_grupo < 0:
                vl_total_veiculos_pesados_grupo = 0
            if vl_total_veiculos_pesados_grupo > 9782:
                vl_total_veiculos_pesados_grupo = 9782
            # quantos são leves
            vl_total_veiculos_leves_grupo = st.number_input('Valor dos veículos leves')
            if vl_total_veiculos_leves_grupo < 0:
                vl_total_veiculos_leves_grupo = 0
            if vl_total_veiculos_leves_grupo > 122090:
                vl_total_veiculos_leves_grupo = 122090            
            
        else:
            fl_veiculo = 0
            vl_total_veiculos_pesados_grupo = 0
            vl_total_veiculos_leves_grupo = 0
            
        # quantidade de meses após atulizar saúde tributária
        nu_meses_rescencia = st.number_input('Quantos meses passaram desde última atualização da saúde tributária?')
        if nu_meses_rescencia < 0:
             nu_meses_rescencia = 0
        if nu_meses_rescencia > 66:
             nu_meses_rescencia = 66    
        
        # Quantidade de coligados
        qt_coligados = st.number_input('Quantos coligados cliente possui?')
        if qt_coligados < 0:
             qt_coligados = 0
        if qt_coligados > 844:
             qt_coligados = 844    
        
        if qt_coligados > 0:
            qt_ufs_coligados = st.number_input('Em quantos estados os coligados atuam?')
            if qt_ufs_coligados < 1:
                qt_ufs_coligados = 1
            if qt_ufs_coligados > 25:
                qt_ufs_coligados = 25    
            qt_ramos_coligados = st.number_input('Em quantos ramos os coligados atuam?')
            if qt_ramos_coligados < 1:
                qt_ramos_coligados = 1
            if qt_ramos_coligados > 86:
                qt_ramos_coligados = 86    
            media_vl_folha_coligados = st.number_input('Qual valor médio da folha de pagamento dos coligados?')
            if media_vl_folha_coligados < .01:
                media_vl_folha_coligados = .01
            if media_vl_folha_coligados > 1205969000000000:
                media_vl_folha_coligados = 1205969000000000
            media_vl_folha_coligados_gp = st.number_input('Qual valor médio da folha de pagamento dos coligados do grupo?')
            if media_vl_folha_coligados_gp < .01:
                media_vl_folha_coligados_gp = .01
            if media_vl_folha_coligados_gp > 24611440000000000:
                media_vl_folha_coligados_gp = 24611440000000000
            media_faturamento_est_coligados = st.number_input('Qual valor médio do faturamento dos coligados?')
            if media_faturamento_est_coligados < .01:
                media_faturamento_est_coligados = .01
            if media_faturamento_est_coligados > 55768500000000000:
                media_faturamento_est_coligados = 55768500000000000       
        else:
            qt_ufs_coligados = 0
            qt_ramos_coligados = 0
            media_vl_folha_coligados = 0
            media_vl_folha_coligados_gp = 0
            media_faturamento_est_coligados = 0
            
        # quantidade coligadas
        qt_coligadas = st.number_input('Quantas coligadas cliente possui?')
        if qt_coligadas < 0:
             qt_coligadas = 0
        if qt_coligadas > 761:
             qt_coligadas = 761    
        
        if qt_coligadas > 0:
            sum_faturamento_estimado_coligadas = st.number_input('Qual soma do faturamento das coligadas?')
            if sum_faturamento_estimado_coligadas < 0:
                sum_faturamento_estimado_coligadas = 0
            if sum_faturamento_estimado_coligadas > 56013370000000000:
                sum_faturamento_estimado_coligadas = 56013370000000000
        else:
            sum_faturamento_estimado_coligadas = 0
            
        # faturamento total
        vl_faturamento_estimado_grupo_aux = st.number_input('Soma faturamento matriz e ramificações')
        if vl_faturamento_estimado_grupo_aux < 0:
            vl_faturamento_estimado_grupo_aux = 0
        if vl_faturamento_estimado_grupo_aux > 222761800000000000:
            vl_faturamento_estimado_grupo_aux = 222761800000000000
        
        # qtde filiais
        qt_filiais = st.number_input('Quantas filiais possui?')
        if qt_filiais < 0:
            qt_filiais = 0
        if qt_filiais > 761:
            qt_filiais = 761
        
        # qtde sócios PJ
        qt_total_socios_pj = st.number_input('Quantos sócios PJ possui?')
        if qt_total_socios_pj < 0:
            qt_total_socios_pj = 0
        if qt_total_socios_pj > 13:
            qt_total_socios_pj = 13
        
        # indicador de saude tributária
        de_saude_tributaria = st.selectbox('Saúde tributária', 
                                             ['VERDE', 'CINZA', 'AMARELO', 'LARANJA', 'AZUL', 'VERMELHO'])
        de_saude_tributaria_VERDE = 0
        de_saude_tributaria_CINZA = 0
        de_saude_tributaria_AMARELO = 0
        de_saude_tributaria_LARANJA = 0
        de_saude_tributaria_AZUL = 0
        de_saude_tributaria_VERMELHO = 0
        if de_saude_tributaria == 'VERDE':
            de_saude_tributaria_VERDE = 1
        elif de_saude_tributaria == 'CINZA':
            de_saude_tributaria_CINZA = 1
        elif de_saude_tributaria == 'AMARELO':
            de_saude_tributaria_AMARELO = 1        
        elif de_saude_tributaria == 'LARANJA':
            de_saude_tributaria_LARANJA = 1   
        elif de_saude_tributaria == 'AZUL':
            de_saude_tributaria_AZUL = 1   
        elif de_saude_tributaria == 'VERMELHO':
            de_saude_tributaria_VERMELHO = 1  
            
        # Nível de atividade
        de_nivel_atividade = st.selectbox('Indique o nível de atividade', 
                                             ['ALTA', 'BAIXA', 'MEDIA'])
        de_nivel_atividade_ALTA = 0
        de_nivel_atividade_BAIXA = 0
        de_nivel_atividade_MEDIA = 0
        if de_nivel_atividade == 'ALTA':
            de_nivel_atividade_ALTA = 1
        elif de_nivel_atividade == 'BAIXA':
            de_nivel_atividade_BAIXA = 1
        elif de_nivel_atividade == 'MEDIA':
            de_nivel_atividade_MEDIA = 1        
        
        # Saúde rescência
        de_saude_rescencia = st.selectbox('Informe o tempo de saúde rescência', 
                                             ['ATE 6 MESES', 'ATE 1 ANO', 'ACIMA DE 1 ANO', 'NENHUMA DAS OPÇÕES'])
        de_saude_rescencia_ACIMA_DE_1_ANO = 0
        de_saude_rescencia_ATE_1_ANO = 0
        de_saude_rescencia_SEM_INFORMACAO = 0
        de_saude_rescencia_ATE_6_MESES = 0
        if de_saude_rescencia == 'ATE 6 MESES':
            de_saude_rescencia_ATE_6_MESES = 1
        elif de_saude_rescencia == 'ATE 1 ANO':
            de_saude_rescencia_ATE_1_ANO = 1
        elif de_saude_rescencia == 'ACIMA DE 1 ANO':    
            de_saude_rescencia_ACIMA_DE_1_ANO = 1
        elif de_saude_rescencia == 'NENHUMA DAS OPÇÕES':    
            de_saude_rescencia_SEM_INFORMACAO = 1    
            
        # criar dicionário de dados:
        dict_dados = {'sg_uf': sg_uf,
        'idade_empresa_anos': [np.float64(idade_empresa_anos)],
        'fl_email': [np.int64(fl_email)],
        'fl_telefone': [np.int64(fl_telefone)],
        'fl_rm': [np.int64(fl_rm)],
        'fl_veiculo': [np.float64(fl_veiculo)],
        'vl_total_veiculos_pesados_grupo': [np.float64(vl_total_veiculos_pesados_grupo)],
        'vl_total_veiculos_leves_grupo': [np.float64(vl_total_veiculos_leves_grupo)],
        'nu_meses_rescencia': [np.float64(nu_meses_rescencia)],
        'qt_coligados': [np.float64(qt_coligados)],
        'qt_ufs_coligados': [np.float64(qt_ufs_coligados)],
        'qt_ramos_coligados': [np.float64(qt_ramos_coligados)],
        'media_vl_folha_coligados': [np.float64(media_vl_folha_coligados)],
        'media_vl_folha_coligados_gp': [np.float64(media_vl_folha_coligados_gp)],
        'media_faturamento_est_coligados': [np.float64(media_faturamento_est_coligados)],
        'qt_coligadas': [np.float64(qt_coligadas)],
        'sum_faturamento_estimado_coligadas': [np.float64(sum_faturamento_estimado_coligadas)],
        'vl_faturamento_estimado_grupo_aux': [np.float64(vl_faturamento_estimado_grupo_aux)],
        'qt_filiais': [np.int64(qt_filiais)],
        'qt_total_socios_pj': [np.float64(qt_total_socios_pj)],
        'de_saude_tributaria_VERDE': [np.int64(de_saude_tributaria_VERDE)],
        'de_saude_tributaria_CINZA': [np.int64(de_saude_tributaria_CINZA)],
        'de_saude_tributaria_AMARELO': [np.int64(de_saude_tributaria_AMARELO)],
        'de_saude_tributaria_LARANJA': [np.int64(de_saude_tributaria_LARANJA)],
        'de_saude_tributaria_AZUL': [np.int64(de_saude_tributaria_AZUL)],
        'de_saude_tributaria_VERMELHO': [np.int64(de_saude_tributaria_VERMELHO)],
        'de_nivel_atividade_ALTA': [np.int64(de_nivel_atividade_ALTA)],
        'de_nivel_atividade_BAIXA': [np.int64(de_nivel_atividade_BAIXA)],
        'de_nivel_atividade_MEDIA': [np.int64(de_nivel_atividade_MEDIA)],
        'de_saude_rescencia_ACIMA DE 1 ANO': [np.int64(de_saude_rescencia_ACIMA_DE_1_ANO)],
        'de_saude_rescencia_ATE 1 ANO': [np.int64(de_saude_rescencia_ATE_1_ANO)],
        'de_saude_rescencia_SEM INFORMACAO': [np.int64(de_saude_rescencia_SEM_INFORMACAO)],
        'de_saude_rescencia_ATE 6 MESES': [np.int64(de_saude_rescencia_ATE_6_MESES)]}
        
        # criar dataframe a partir do dicionário
        df = pd.DataFrame(data=dict_dados)
        
        # fazer merge com os dados dos indicadores econômico-sociais
        df_ind_socio_eco = pd.read_excel('data/dados_economicos.xls')
        df_ind_socio_eco.rename(columns={'UF': 'sg_uf'}, inplace=True)
        df.set_index('sg_uf', inplace=True)
        df_ind_socio_eco.set_index('sg_uf', inplace=True)
        df = df.join(df_ind_socio_eco)
        df.reset_index(inplace=True)
        # Gerar lista de variáveis numéricas
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        ls_var_numerica = list(df.select_dtypes(include=numerics).columns)
        
        # escalar os dados com min_max_scaler das análises feitas com notebooks
        from sklearn.preprocessing import MinMaxScaler
        scaler = joblib.load('objeto/scaler.pkl')
        df[ls_var_numerica] = scaler.transform(df[ls_var_numerica])
        
        # carregar o modelo e fazer a predição
        from sklearn.cluster import KMeans
        X = df.drop('sg_uf', axis=1)
        kmeans = joblib.load('modelo/kmeans_model.pkl')
        cluster_do_id = kmeans.predict(X)
        distancias_cluster = kmeans.transform(X)
        dist_ao_cluster = np.min(distancias_cluster)
        cluster_predito = cluster_do_id[0]
        # selecionar clientes
        selecionar_clientes(cluster_predito, dist_ao_cluster)
    
    
if __name__ == '__main__':
    main()