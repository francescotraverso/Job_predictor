import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_bert():
    # load saved BERT model and its embedding from disk

    read_path = '/'.join(os.path.realpath(__file__).split('/')[0:-3]) + \
        '/model/'

    filename = read_path + 'bert_model.sav'
    bert_model = pickle.load(open(filename, 'rb'))

    filename = read_path + 'all_corpus_embed.sav'
    all_corpus_embed = pickle.load(open(filename, 'rb'))

    return bert_model, all_corpus_embed


def run_bert(
        bert_model,
        all_corpus_embed,
        new_description,
        df_occ_n_skills,
        n_jobs=5):

    # run the preloaded BERT model
    new_description_embed = bert_model.encode(new_description)

    # calculate and order cosine similarity
    sim_rank = cosine_similarity(
        [new_description_embed],
        all_corpus_embed)

    sim_rank_ind = np.argsort(sim_rank[0])[::-1]

    # print model results
    print(f'TEST DOCUMENT: {new_description} \n')
    print(f'LISTING {n_jobs} MOST SIMILAR JOB ROLES & DESCRIPTIONS \n')

    for i in range(n_jobs):

        if sim_rank_ind[i] <= len(df_occ_n_skills):
            new_index = sim_rank_ind[i]
        else:
            new_index = sim_rank_ind[i] - len(df_occ_n_skills)

        original_job_title = df_occ_n_skills.loc[new_index]['job_title']

        if 'senior' in original_job_title:
            original_job_title = original_job_title.replace('senior ', '')

        print(f'RANK #{i+1}: ' + original_job_title)
        # print(df_occ_n_skills.loc[new_index]['description'])    don't show JD
        print(f'Similarity score: \
            {round(sim_rank[0][sim_rank_ind[i]]*100,1)} % \n')

def run_bert_api(
        bert_model,
        all_corpus_embed,
        new_description,
        df_occ_n_skills,
        n_jobs=5):

    # run the preloaded BERT model
    new_description_embed = bert_model.encode(new_description)

    # calculate and order cosine similarity
    sim_rank = cosine_similarity(
        [new_description_embed],
        all_corpus_embed)

    sim_rank_ind = np.argsort(sim_rank[0])[::-1]

    api_output_dict = {}

    for i in range(n_jobs):

        if sim_rank_ind[i] <= len(df_occ_n_skills):
            new_index = sim_rank_ind[i]
        else:
            new_index = sim_rank_ind[i] - len(df_occ_n_skills)

        original_job_title = df_occ_n_skills.loc[new_index]['job_title']

        if 'senior' in original_job_title:
            original_job_title = original_job_title.replace('senior ', '')

        dict_key = 'Job #' + str(i + 1)
        api_output_dict[dict_key] = { \
            'Title': original_job_title,
            'Proximity Score': round(sim_rank[0][sim_rank_ind[i]]*100,1)}

    return api_output_dict
