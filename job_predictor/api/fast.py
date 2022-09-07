from job_predictor.ml_logic.data import get_occupation_df, open_area_kws
from job_predictor.ml_logic.preprocessor import preprocess_input
from job_predictor.ml_logic.preprocessor import truncate_description
from job_predictor.ml_logic.model import load_bert, run_bert_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.state.model = load_bert()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
"""
# replace this with user dropdown menu choice
area_choice = 'ICT

# replace with user CV
describe_your_job = '
Management Consulting in FREESCG - Food Consultant - Management consultancy specialized in food & beverage - Consulting services for restaurants, bars and similar - Project developed with specialization to each client through a diagnosis and an action plan put into practice, with analysis of the obtained results - Areas of
'
"""

@app.get("/job_titles")
def job_titles(describe_your_job, area_choice):

    df_occ_n_skills = get_occupation_df()

    area_keywords = open_area_kws(area_choice)

    new_description = truncate_description(
        describe_your_job,
        no_words=100)

    new_description = preprocess_input(
        new_description,
        area_keywords,
        area_kw_insert=True,
        area_kw_insert_ratio=0.3)

    bert_model, all_corpus_embed = load_bert()

    api_output_dict = run_bert_api(
                            bert_model,
                            all_corpus_embed,
                            new_description,
                            df_occ_n_skills)


    return api_output_dict


@app.get("/")
def root():
    return {'greeting': 'Hello'}
