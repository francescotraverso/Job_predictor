import pandas as pd
import csv
import os


def get_occupation_df():

    read_path = '/'.join(os.path.realpath(__file__).split('/')[0:-3])

    # fetch dataframes from CSV files
    df_occ_n_skills = pd.read_csv(read_path + \
        '/data/ESCO/occupations_augmented_with_OPTIONAL_SKILLS.csv',
        verbose=False)

    # filter out unnecessary columns
    df_occ_n_skills = df_occ_n_skills.filter(
        items=['preferredLabel', 'description', 'skills', 'opt_skills'])
    df_occ_n_skills = df_occ_n_skills.reindex(
        columns=['preferredLabel','description', 'skills', 'opt_skills'])

    # make column names more readable
    df_occ_n_skills.rename(
        columns={'preferredLabel': 'job_title'}, inplace=True)

    # get ready to receive modellable columns
    df_occ_n_skills['description_input'] = 0
    df_occ_n_skills['skills_input'] = 0

    # create description_input and skills_input,
    # which are strings on which the model will be fit
    for row, index in df_occ_n_skills.iterrows():
        underscored_job_title = index['job_title'].replace(" ", "_")
        this_rows_description_input = underscored_job_title + ' ' + \
            index['description']
        if type(index['opt_skills']) != str:
            index['opt_skills'] = ''
        this_rows_skills_input = underscored_job_title + ' ' + \
            index['skills'] + ', ' + index['opt_skills']
        df_occ_n_skills.iloc[row,-2] = this_rows_description_input
        df_occ_n_skills.iloc[row,-1] = this_rows_skills_input

    return df_occ_n_skills


def open_area_kws(area_choice):

    flare_dict = {
        'management': 'management_skills_esco.csv',
        'sales': 'sales_skills_esco.csv',
        'tourism': 'tourism_skills_esco.csv',
        'data science': 'data_science_skills_esco.csv',
        'ICT': 'ict_skills_esco.csv',
        'marketing': 'marketing_skills_esco.csv',
    }

    read_path = '/'.join(os.path.realpath(__file__).split('/')[0:-3]) + \
        '/data/skill_packages/ESCO/'

    with open(read_path + flare_dict[area_choice], newline='') as f:
        reader = csv.reader(f)
        csv_data = list(reader)
        area_keywords = [item for sublist in csv_data for item in sublist]

    return area_keywords
