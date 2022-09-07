from job_predictor.ml_logic.data import get_occupation_df, open_area_kws
from job_predictor.ml_logic.preprocessor import preprocess_input
from job_predictor.ml_logic.preprocessor import truncate_description
from job_predictor.ml_logic.model import load_bert, run_bert

# replace this with user dropdown menu choice
area_choice = 'ICT'

# replace with user CV
describe_your_job = """
Management Consulting in FREESCG - Food Consultant - Management consultancy specialized in food & beverage - Consulting services for restaurants, bars and similar - Project developed with specialization to each client through a diagnosis and an action plan put into practice, with analysis of the obtained results - Areas of
"""

df_occ_n_skills = get_occupation_df()

area_keywords = open_area_kws(area_choice)

new_description = truncate_description(
    describe_your_job,
    no_words=100)

new_description = preprocess_input(
    new_description,
    area_keywords,
    area_kw_insert=True,
    area_kw_insert_ratio=0.2)

bert_model, all_corpus_embed = load_bert()

run_bert(
    bert_model,
    all_corpus_embed,
    new_description,
    df_occ_n_skills,
    n_jobs=10)
