import pickle
from bson import ObjectId
from datetime import datetime, timedelta
import pandas as pd
import xgboost as xgb

# define existence of mlAritact of user function
def account_artifact_checker(mongo_client,
                             src_database_name: str,
                             src_collection_name: str, 
                             account_id):
    
    # find for ml artifact of the account and return boolean of existence
    existence = mongo_client[src_database_name][src_collection_name].find({'account': account_id})
    
    return existence

# define function to get country code of the account
def get_country_code(mongo_client,
                     account_id,
                     app_db: str,
                     account_collection: str):
    
    temp = mongo_client[app_db][account_collection].find({'_id': account_id}, 
                                                         {'_id': 0,
                                                          'countryCode':'$geolocation.countryCode'})
    country_code = temp[0]['countryCode']
    
    return country_code

# define load model artifact to database function
def load_model_from_mongodb(mongo_client,
                            src_database_name: str,
                            src_collection_name: str, 
                            model_name: str, 
                            account_id):
    
    json_data = {} # pre-define model as json format
    
    # find model as corresponding user id #! will change to account id in the future
    data = mongo_client[src_database_name][src_collection_name].find({
                                                                      'account': account_id,
                                                                      'model': model_name
    })

    # loop throgh schema
    for i in data:
        
        json_data = i
        
    # get model artifact
    pickled_model = json_data['artifact']

    return pickle.loads(pickled_model)

# define feature preparation function from content id list
def prepare_features(mongo_client,
                     content_id_list, 
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str):
    
    # define cursor of content features
    contentFeaturesCursor = [
        {
            # filter for only correspond contents
            '$match': {
                'contentId': {
                    '$in': content_id_list 
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
                'contentId': 1,
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
    # alias 'contentFeatures_1'
    content_features = pd.DataFrame(list(mongo_client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
    
    return content_features


# define function to formating output
def convert_lists_to_dict(contents_id_list, 
                          prediction_scores):
    
    result = {}
    
    for index, _ in enumerate(prediction_scores):
    
        result[contents_id_list[index]] = prediction_scores[index]
    
    return result


# define main function
def personalized_content_predict_main(event,
                                      mongo_client,
                                      src_database_name: str,
                                      src_collection_name: str,
                                      analytics_db: str,
                                      app_db: str,
                                      account_collection: str,
                                      ml_arifact_country_collection: str,
                                      creator_stats_collection: str,
                                      content_stats_collection: str,
                                      model_name: str):
    
    # 1. get input
    #! convert to object id
    account_id = ObjectId(event.get('accountId', None))
    print("accountId:", account_id)
    
    #! convert to object id
    content_id_list = [ObjectId(content) for content in event.get('contents', None)]
    print("content_list:", content_id_list)
    print("len content_list:",len(content_id_list))

    # check existence of personalize content artifact of the account 
    existence = account_artifact_checker(mongo_client,
                                         src_database_name=src_database_name,
                                         src_collection_name=src_collection_name, 
                                         account_id=account_id)
    print("existence:", existence)
    
    # 2. loading model
    # case mlArtifacts exists
    if len(list(existence)) != 0: #! in testing, use ""== 0" in deployment use "!= 0"
        
        #!
        print('case: mlArtifact exists')
        print('this comes from existence = true')
        
        # perform model loading function
        xg_reg = load_model_from_mongodb(mongo_client,
                                         src_database_name=src_database_name,
                                         src_collection_name= src_collection_name,
                                         model_name= model_name,
                                         account_id=account_id)
    
    # case mlArtifacts does not exists, the model come from coldstart
    else:
        
        #!
        print('case: mlArtifact not exists')
        print('this comes from existence = false')
        
        # get country code
        country_code = get_country_code(mongo_client=mongo_client, account_id=account_id, app_db=app_db, account_collection=account_collection)

        #!
        print(country_code)    
        
        # perform model loading function
        xg_reg = load_model_from_mongodb(mongo_client,
                                         src_database_name=src_database_name,
                                         src_collection_name= ml_arifact_country_collection,
                                         model_name= model_name,
                                         account_id=country_code)

    # 3. preparation
    # prepare_features
    content_features = prepare_features(mongo_client,
                                        content_id_list,
                                        analytics_db = analytics_db,
                                        content_stats_collection = content_stats_collection,
                                        creator_stats_collection = creator_stats_collection)
    print("content_features:", content_features)
    print("len content_features:", len(content_features))
    
    # 4. prediction
    # define result format
    prediction_scores = [float(score) for score in (xg_reg.predict(content_features.drop('contentId', axis = 1)))]
    print("prediction_scores:", prediction_scores)
    print("len prediction_scores:", len(prediction_scores))
    
    
    # 5. construct result schemas
    result = convert_lists_to_dict(contents_id_list = content_id_list), 
                                   prediction_scores = prediction_scores)
    # result = convert_lists_to_dict(contents_id_list = event.get('contents', None), 
                                #    prediction_scores = prediction_scores)

    print('final length of scores:')
    print(len(prediction_scores))
    print(prediction_scores)
    print('len content id list:')
    print(len(content_id_list))

    response = {
        'statusCode': 200,
        'result': result
    }

    print('response:')
    print(len(response['result']))
    print(response['result'])
    print('result:')
    print(len(result))
    print(result)
    
    return response
    