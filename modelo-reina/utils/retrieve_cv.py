import os
import requests
import pandas as pd

# url = 'https://services6.arcgis.com/POowwbv4rcaNpUgV/arcgis/rest/services/acumulados_comunitatvalenciana' \
#       '/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*' \
#       '&orderByFields=Data%20desc&resultOffset=0&resultRecordCount=365&resultType=standard&cacheHint=true'
#
# data = requests.get(url).json()
# features = [att['attributes'] for att in data['features']]
# df = pd.DataFrame(features).drop(columns=['CCAA_Codigo_ISO', 'Data', 'J', 'FID', ]).rename(columns={'I': 'Data'})
# df.to_csv('cv_retrieved_daily.csv', index=None)


url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/3/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&resultOffset=0&resultRecordCount=1000'
data = requests.get(url).json()
features = [att['attributes'] for att in data['features']]
df = pd.DataFrame(features).drop(columns=['ccaa_codigo_iso', 'taxa', 'taxa_creixement', 'a', 'b', 'c', 'h']).rename(columns={'data': 'timestamp', 'i': 'Data'})
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df = df.sort_values(by=['Data'])
df = df.drop(labels=['objectid', 'timestamp', 'd', 'e', 'f', 'g', 'j'], axis=1)
df.to_csv(os.path.join('..', 'data', 'cv_retrieved_daily.csv'), index=None)

# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/1/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=depart%20asc&resultOffset=0&resultRecordCount=25'
# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&groupByFieldsForStatistics=edad%2Csexo&orderByFields=edad%20asc&outStatistics=%5B%7B%22statisticType%22%3A%22sum%22%2C%22onStatisticField%22%3A%22posit%22%2C%22outStatisticFieldName%22%3A%22value%22%7D%5D'
# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&groupByFieldsForStatistics=edad%2Csexo&orderByFields=edad%20asc&outStatistics=%5B%7B%22statisticType%22%3A%22sum%22%2C%22onStatisticField%22%3A%22fallec%22%2C%22outStatisticFieldName%22%3A%22value%22%7D%5D'
# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&groupByFieldsForStatistics=edad%2Csexo&orderByFields=edad%20asc&outStatistics=%5B%7B%22statisticType%22%3A%22sum%22%2C%22onStatisticField%22%3A%22porc_posit%22%2C%22outStatisticFieldName%22%3A%22value%22%7D%5D'
# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects'
# url = 'https://carto.icv.gva.es/arcgis/rest/services/covid19/Covid19_Monitorizacion_sde_v6/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&groupByFieldsForStatistics=edad%2Csexo&orderByFields=edad%20asc&outStatistics=%5B%7B%22statisticType%22%3A%22sum%22%2C%22onStatisticField%22%3A%22porc_fall%22%2C%22outStatisticFieldName%22%3A%22value%22%7D%5D'


