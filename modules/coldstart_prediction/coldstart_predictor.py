def cold_start_by_counytry_scroing( client,
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'saved_prediction_country',
                                    saved_data_all = 'saved_prediction_country_accum',
                                    model_name = 'xgboost'):
    
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

    def prepare_features(mongo_client, 
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str):
    
    # define cursor of content features
        contentFeaturesCursor = [
         {
            # join with creator stats
                '$lookup': {
                    'from': creator_stats_collection, # previous:'creatorStats',
                    'localField': 'authorId',
                    'foreignField': '_id',
                    'as': 'creatorStats'
                }
            }, {
            # deconstruct array
                '$unwind': {
                    'path': '$creatorStats',
                    'preserveNullAndEmptyArrays': True
                }
            }, {
            # map output format
                '$project': {
                    '_id': 1,
                    'likeCount': 1,
                    'commentCount': 1,
                    'recastCount': 1,
                    'quoteCount': 1,
                    'photoCount': 1,
                    'characterLength': 1,
                    'creatorContentCount' :'$creatorStats.contentCount',
                    'creatorLikedCount': '$creatorStats.creatorLikedCount',
                    'creatorCommentedCount': '$creatorStats.creatorCommentedCount',
                    'creatorRecastedCount': '$creatorStats.creatorRecastedCount',
                    'creatorQuotedCount': '$creatorStats.creatorQuotedCount',
                    'ageScore': '$aggregator.ageScore'
#                 # alias 'total label'
#                 'engagements': {
#                     '$sum': [
#                         '$likeCount', 
#                         '$commentCount',
#                         '$recastCount',
#                         '$quoteCount'
#                     ]
#                 }
                }
            }
        ]

    # assign result to dataframe
    # alias 'contentFeatures_1'

        content_features = pd.DataFrame(list(client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
    
        return content_features

    contentFeatures = prepare_features(mongo_client = client, # default
                                        analytics_db = 'analytics-db',
                                        content_stats_collection = 'contentStats',
                                        creator_stats_collection = 'creatorStats')

    mlArtifacts_country = analyticsDb[saved_model]
    ml_set = pd.DataFrame(list(mlArtifacts_country.find()))
    
    saved_data_country = appDb[saved_data]
    
    saved_data_country_accum = analyticsDb[saved_data_all]
    
    saved_data_country.remove({})
    saved_data_country_accum.remove({})
    
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

        content_test = contentFeatures
    
        a = pd.DataFrame(xg_reg_load.predict(content_test.drop(['contentId'], axis = 1)), columns = ['predict'])
        b = contentFeatures[['contentId']].reset_index(drop = True)
        c = pd.concat([b,a],axis =1)
        c['countryCode'] = countryId
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
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'guestfeeditems',
                                    saved_data_all = 'saved_prediction_country_accum',
                                    model_name = 'xgboost')
    
    return