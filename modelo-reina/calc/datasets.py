import pandas as pd
from calc import calcfunc, get_root_path


@calcfunc()
def get_population(area_name):
    df = None
    if area_name == 'Comunitat Valenciana':
        df = pd.read_csv(get_root_path() + '/data/cv_pop.csv')
    elif area_name == 'CVDebug1000':
        df = pd.read_csv(get_root_path() + '/data/cv_pop.csv') // 1000
    return df


@calcfunc(variables=['country', 'max_age'])
def get_contacts_for_country(variables):
    f = open(get_root_path() + '/data/contact_matrix.csv', 'r')
    max_age = variables['max_age']

    df = pd.read_csv(f, header=0)
    df = df[df.country == variables['country']].drop(columns='country')

    df['place_type'] = df['place_type'].map(lambda x: x.replace('cnt_', '').replace('otherplace', 'other'))
    s = '-%d' % max_age
    df['participant_age'] = df['participant_age'].map(lambda x: x.replace('+', s))
    last_col = [x for x in df.columns if '+' in x]
    assert len(last_col) == 1
    df = df.rename(columns={last_col[0]: last_col[0].replace('+', s)})

    return df


@calcfunc()
def get_healthcare_districts():
    p = get_root_path() + '/data/shp_jasenkunnat_2020.xls'
    df = pd.read_excel(p, header=3, sheet_name='shp_jäsenkunnat_2020_lkm')
    df = df[['kunta', 'sairaanhoitopiiri', 'erva-alue']].dropna()
    return df


@calcfunc(variables=['area_name'])
def get_population_for_area(variables):
    df = get_population(variables['area_name'])
    return df

