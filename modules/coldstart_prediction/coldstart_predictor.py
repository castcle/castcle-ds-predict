def cold_start_by_counytry_scroing( client,
                                    saved_model = 'mlArtifacts_country_test',
                                    saved_data = 'saved_prediction_country',
                                    saved_data_all = 'saved_prediction_country_accum',
                                    content_features = 'contentFeatures',
                                    model_name = 'xgboost'):
    
    import pymongo # connect to MongoDB
    from pymongo import MongoClient # client connection to MongoDB
    import sklearn
    import pandas as pd
    import json
    import xgboost as xgb
    import bson.objectid
    import pickle
    from datetime import datetime
    from pprint import pprint
    import numpy as np

    appDb = client['app-db']
    analyticsDb = client['analytics-db']
 
    contentFeatures = analyticsDb[content_features]
    contentFeatures = pd.DataFrame(list(contentFeatures.find()))

    mlArtifacts_country = analyticsDb[saved_model]
    ml_set = pd.DataFrame(list(mlArtifacts_country.find()))
    
    saved_data_country = analyticsDb[saved_data]
    
    saved_data_country_accum = analyticsDb[saved_data_all]
    
    saved_data_country.drop()
    saved_data_country_accum.drop()
    
    def load_model_from_mongodb(collection, model_name, account):
        json_data = {}
        data = collection.find({
            'account': account,
            'model': model_name
        })
    
        for i in data:
            json_data = i
    
        pickled_model = json_data['artifact']
    
        return pickle.loads(pickled_model)
    
    result = pd.DataFrame()
    for countryId in list(ml_set.account.unique()):
        pprint(countryId)
        xg_reg_load = load_model_from_mongodb(collection=mlArtifacts_country,
                                     account= countryId,
                                     model_name= model_name)

        content_test = contentFeatures.drop(['userId'], axis = 1)
    
        a = pd.DataFrame(xg_reg_load.predict(content_test.drop(['_id'], axis = 1)), columns = ['predict'])
        b = contentFeatures[['_id']].reset_index(drop = True)
        c = pd.concat([b,a],axis =1).rename({'_id':'contentId'},axis = 1)
        c['country_code'] = countryId
        c['Score_At'] = datetime.now() 
        c = c.sort_values(by='predict', ascending=False)
        c = c.iloc[:2000,]
        result = result.append(c)  

        
     # update collection
    result.reset_index(inplace=False)
    data_dict = result.to_dict("records")
    
    saved_data_country.insert_many(data_dict)
    
    saved_data_country_accum.insert_many(data_dict)
    
#saved_data_country.update_one({'countryId': countryId},{'$set':{"scoring_list":data_dict}},upsert= True)

    return
    
def coldstart_score_main(client):
    cold_start_by_counytry_scroing( client, 
                                    saved_model = 'mlArtifacts_country_test',
                                    saved_data = 'saved_prediction_country',
                                    saved_data_all = 'saved_prediction_country_accum',
                                    content_features = 'contentFeatures',
                                    model_name = 'xgboost')
    

    
    return