import os
import json
import pickle
import bson.objectid
from bson import ObjectId
from datetime import datetime, timedelta
import pandas as pd
import xgboost as xgb

# define load model artifact to database function
def load_model_from_mongodb(mongo_client, 
                            src_database_name: str,
                            src_collection_name: str, 
                            model_name: str, 
                            account_id, #! using user id, in future version change to account id
                            ):
    
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
def prepare_features( 
                     content_id_list,
                     mongo_client,
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str):
    
    # define cursor of content features
    contentFeaturesCursor = [
        {
            # filter for only correspond contents
            '$match': {
                '_id': {
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
    content_features = pd.DataFrame(
        list(mongo_client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor)))\
            .rename({'_id':'contentId'},axis = 1)
    
    return content_features

# define save feed item function
def save_feed_to_mongodb(mongo_client, 
                         user_id,
                         content_id_list,
                         prediction_score,
                         dst_database_name: str,
                         dst_collection_name: str):


    document = mongo_client[dst_database_name][dst_collection_name].update_one(
        {
            'viewer': user_id
        }, {
            '$set': {
                'viewer': user_id,
                'contents': content_id_list,
                'prediction_score': prediction_score,
                'scoredAt': datetime.utcnow()
            }
        }, upsert= True)

    return None

# define main function
def personalized_content_predict_main(event,
                                      mongo_client,
                                      src_database_name = 'analytics-db',
                                      src_collection_name = 'mlArtifacts',
                                      analytics_db = 'analytics-db',
                                      creator_stats_collection = 'creatorStats',
                                      content_stats_collection = 'contentStats',
                                      dst_database_name = 'analytics-db',
                                      dst_collection_name = 'feedItems_test',
                                      model_name = 'xgboost'):
    
    # 1. get input
    #! convert to object id
    user_id = ObjectId(event.get('userId', None))
    
    #! convert to object id
    contents_list = event.get('contents', None)
    # If has contents
    if len(contents_list) > 0:
    	content_id_list = [ObjectId(content) for content in contents_list]
    # If no contents with this userId
    elif len(contents_list) == 0:
        raise NotImplementedError
    
    # 2. loading model
    # perform model loading function
    xg_reg = load_model_from_mongodb(mongo_client=mongo_client, 
                                     src_database_name=src_database_name,
                                     src_collection_name= src_collection_name,
                                     model_name= model_name,
                                     account_id=user_id) # tend to change name
    
    # 3. preparation
    # prepare_features
    content_features = prepare_features(
        			   content_id_list, 
			 		   mongo_client=mongo_client,
                       analytics_db = analytics_db,
                       content_stats_collection = content_stats_collection,
                       creator_stats_collection = creator_stats_collection)
    if 'contentId' in content_features.columns:
    	content_features = content_features.drop('contentId', axis = 1)
    
    # 4. prediction
    # define result format
    prediction_score = [float(score) for score in (xg_reg.predict(content_features))]
    
    # 5. save result
    #! upsert results to destination collection
    save_feed_to_mongodb(user_id,
                         content_id_list,
                         prediction_score,
                         dst_database_name=dst_database_name,
                         dst_collection_name=dst_collection_name)

    
    return None