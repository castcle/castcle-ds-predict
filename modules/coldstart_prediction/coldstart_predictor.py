
# main function of personalize content model prediction
# 1. get input
# 2. loading model
# 3. feature preparation
# 4. model prediction
# 5. construct result schemas

def cold_start_by_counytry_scroing( client,
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'guestfeeditems',
                                    saved_data_temp = 'guestfeeditemstemps',
                                    model_name = 'xgboost',
                                    updatedAtThreshold = 30.0):
    
    import pandas as pd
    import pickle
    from datetime import datetime
    from pprint import pprint
    from datetime import datetime, timedelta

    # connect to database
    appDb = client['app-db']
    analyticsDb = client['analytics-db']

    # define feature preparation function from content id list
    def prepare_features(client, 
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str,
                     updatedAtThreshold = updatedAtThreshold):
    
    # define cursor of content features
        contentFeaturesCursor = [
               {
            # filter age of contents for only newer than specific days
                '$match': {
                    'updatedAt': {
                        '$gte': (datetime.utcnow() - timedelta(days=updatedAtThreshold))
                    }
                }
            }, {
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

                }
            }
        ]
    # assign result to dataframe
    # alias 'contentFeatures'

        content_features = pd.DataFrame(list(client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
    
        return content_features

    contentFeatures = prepare_features(client = client, # default
                                        analytics_db = 'analytics-db',
                                        content_stats_collection = 'contentStats',
                                        creator_stats_collection = 'creatorStats',
                                        updatedAtThreshold = updatedAtThreshold)
    # connect to needed collections
    mlArtifacts_country = analyticsDb[saved_model]
    artifact_list = pd.DataFrame(list(mlArtifacts_country.find()))
    
    saved_data_country = appDb[saved_data]
    
    saved_data_country_temporary = appDb[saved_data_temp]
    
    # prepare temporary storage
    saved_data_country_temporary.drop({})
    
    # load model by country function
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
    
    result = pd.DataFrame() # storage for result 
    
    # loop for all country list  
    for countryId in list(artifact_list.account.unique()):
        
        pprint(countryId)
        # load model 
        model_load = load_model_from_mongodb(collection=mlArtifacts_country,
                                     account= countryId,
                                     model_name= model_name)

        contentFeatures_for_scoring = contentFeatures
        
        # scoring process
        score = pd.DataFrame(model_load.predict(contentFeatures_for_scoring.drop(['contentId'], axis = 1)), columns = ['score'])
        
        # set up schema
        content_list = contentFeatures[['contentId']].reset_index(drop = True)
        content_score = pd.concat([content_list,score],axis =1)
        content_score['countryCode'] = countryId
        content_score['type'] = "content"
        content_score['updatedAt'] = datetime.utcnow() 
        content_score['createdAt'] = datetime.utcnow() 
        content_score = content_score.rename({"contentId":"content"},axis = 1)
        content_score = content_score.sort_values(by='score', ascending=False)
        content_score = content_score.iloc[:2000,]
        
        # append result
        result = result.append(content_score)  

        
     # update collection
    result.reset_index(inplace=False)
    
    data_dict = result.to_dict("records")
    
    # save to temporary storage
    saved_data_country_temporary.insert_many(data_dict)
    print('done_save')

    # save to target stoage
    saved_data_country_temporary.rename(saved_data, dropTarget = True)
    print('done_move')
    
def coldstart_score_main(client):
    
    cold_start_by_counytry_scroing( client,
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'guestfeeditems',
                                    saved_data_temp = 'guestfeeditemstemp',
                                    model_name = 'xgboost',
                                    updatedAtThreshold = 30.0)
    

    
    return