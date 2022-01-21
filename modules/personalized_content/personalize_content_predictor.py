'''
main function of personalize content model prediction
1. check database
2. get input
3. loading model
4. feature preparation
5. model prediction
6. construct result schemas
'''
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
    
    '''
    check existence of the user's model artifact
    '''

    # find for ml artifact of the account and return boolean of existence
    existence = mongo_client[src_database_name][src_collection_name].find({'account': account_id})
    
    return existence

# define function to get country code of the account
def get_country_code(mongo_client,
                     account_id,
                     app_db: str,
                     account_collection: str
                     ):

    '''
    find country code from geolocation of the user if absent, default country code "us" will be apply. This function will execute when there is not model artifact
    '''
    
    # geolocation checker
    # case: has geolocation
    if len(list(mongo_client[app_db][account_collection].find({'_id': account_id, 
                                                   'geolocation.countryCode': {'$exists': True}}
                                                  ))) != 0:

        print('user has geolocation')

        # get country code
        temp = mongo_client[app_db][account_collection].find({'_id': account_id}, 
                                                             {'_id': 0,
                                                              'countryCode':'$geolocation.countryCode'})
        
        # assign country code
        country_code = temp[0]['countryCode']
     
    # case does not have geolocation
    else:
        
        
        print('user does not have geolocation')
        
        # set country code to US
        country_code = "us"

    
    return country_code

# define load model artifact to database function
def load_model_from_mongodb(mongo_client,
                            src_database_name: str,
                            src_collection_name: str, 
                            model_name: str, 
                            account_id):

    '''
    retrieve model artifact of the user in case of present or country model artifact in case of model artifact absent
    '''    
    
    json_data = {} # pre-define model as json format
    
    # find model as corresponding account id
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

    '''
    feature preparation from "contentStats" & "creatorStats" for ultilize as feature in model prediction
    '''
    
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
                '_id': 0, # change to 0 from 1
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
                'ageScore': '$aggregator.ageScore',
                'updatedAt': 1
            }
        }
    ]

    # assign result to dataframe
    # alias 'contentFeatures_1'
    content_features = pd.DataFrame(list(mongo_client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
    
    return content_features


# define function to formating output
def convert_lists_to_dict(string_content_id_list, 
                          prediction_scores):

    '''
    convert output (content IDs & prediction scores) into dictionary/json facilitating return response
    '''

    result = {}
    
    for index, _ in enumerate(prediction_scores):
        
        result[string_content_id_list[index]] = prediction_scores[index]
    
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
    import time
    
    '''
    main function of personalize content model prediction
    1. check database
    2. get input
    3. loading model
    4. feature preparation
    5. model prediction
    6. construct result schemas
    '''
    print("=============================================================")
    start = time.time()
    print("[Start] :", start)
    # 1. check database
    t1_start = time.time()
    print("[Start] Check database:", t1_start)

    #mongodb check collection exists
    appDB = mongo_client[app_db]
    anaDB = mongo_client[src_database_name]
    coll_appDB = appDB.list_collection_names()
    coll_anaDB = anaDB.list_collection_names()

    account_collection_return = appDB[account_collection].aggregate([
        {
            "$limit": 1
        }, {
            "$project": {
                "_id": 1
            }
        }
    ])

    content_stats_collection_return = anaDB[content_stats_collection].aggregate([
        {
            "$limit": 1
        }, {
            "$project": {
                "_id": 1
            }
        }
    ])

    creator_stats_collection_return = anaDB[creator_stats_collection].aggregate([
        {
            "$limit": 1
        }, {
            "$project": {
                "_id": 1
            }
        }
    ])

    account_collection_size = len(list(account_collection_return))
    content_stats_collection_size = len(list(content_stats_collection_return))
    creator_stats_collection_size = len(list(creator_stats_collection_return))

    if (account_collection in coll_appDB) and (account_collection_size == 0):
        account_collection_missing = True
    else:
        account_collection_missing = False

    if (content_stats_collection in coll_anaDB) and (content_stats_collection_size == 0):
        content_stats_collection_missing = True
    else:
        content_stats_collection_missing = False

    if (creator_stats_collection in coll_anaDB) and (creator_stats_collection_size == 0):
        creator_stats_collection_missing = True
    else:
        creator_stats_collection_missing = False

    if account_collection_missing \
        or content_stats_collection_missing \
        or creator_stats_collection_missing:

        print('there is no document in', app_db, account_collection)
        print('or', src_database_name, content_stats_collection)
        print('or', src_database_name, creator_stats_collection)
        t1_end = time.time()
        print("[End] If check database fail:", t1_end)
        print("[Time] Check database fail", t1_end - t1_start)
        print("=============================================================")
    else:

        t2_start = time.time()
        print("[Start] If check database success, Getting input:", t2_start)
        # 2. get input
        # convert to object id
        account_id = ObjectId(event.get('accountId', None))
        
        print("accountId:", account_id)
        
        # convert to object id & distinct
        content_id_list = [ObjectId(content) for content in list(set(event.get('contents', None)))]
        
        # define string of content id for response
        string_content_id_list = list(set(event.get('contents', None)))
        
        print('len content id list', len(content_id_list))

        # check existence of personalize content artifact of the account 
        existence = account_artifact_checker(mongo_client,
                                            src_database_name=src_database_name,
                                            src_collection_name=src_collection_name, 
                                            account_id=account_id)

        t2_end = time.time()
        print("[End] Getting input:", t2_end)
        print("[Time] Getting input:", t2_end - t2_start)
        print("=============================================================")
        
        # 3. loading model
        t3_start = time.time()
        print("[Start] Loading model:", t3_start)
        # case mlArtifacts exists
        
        if len(list(existence)) != 0: #! in testing, use ""== 0" in deployment use "!= 0"
            
            #!
            print('case: mlArtifact exists')
            
            # perform model loading function
            xg_reg = load_model_from_mongodb(mongo_client,
                                            src_database_name=src_database_name,
                                            src_collection_name= src_collection_name,
                                            model_name= model_name,
                                            account_id=account_id)

            t3_1_end = time.time()
            print("[End] Artifact exists:", t3_1_end)
            print("[Time] Loading model, artifact exists:", t3_1_end - t3_start)
            print("=============================================================")
        
        # case mlArtifacts does not exists, the model come from coldstart
        else:
            
            #!
            print('case: mlArtifact not exists')
            
            # get country code
            country_code = get_country_code(mongo_client=mongo_client, account_id=account_id, app_db=app_db, account_collection=account_collection)

            #!
            print('country code:', country_code)    
            
            # perform model loading function
            xg_reg = load_model_from_mongodb(mongo_client,
                                            src_database_name=src_database_name,
                                            src_collection_name= ml_arifact_country_collection,
                                            model_name= model_name,
                                            account_id=country_code)

            t3_2_end = time.time()
            print("[End] Artifact not exists:", t3_2_end)
            print("[Time] Loading model, artifact not exists:", t3_2_end - t3_start)
            print("=============================================================")

        t3_end = time.time()
        print("[End] Loading model:", t3_end)
        print("[Time] Loading model", t3_end - t3_start)
        print("=============================================================")
        # 4. feature preparation
        t4_start = time.time()
        print("[Start] Feature preparation:", t4_start)
        # prepare_features
        content_features = prepare_features(mongo_client,
                                            content_id_list,
                                            analytics_db = analytics_db,
                                            content_stats_collection = content_stats_collection,
                                            creator_stats_collection = creator_stats_collection).rename({'updatedAt':'origin'},axis = 1)
        
        print('len of content feature', len(content_features))

        t4_end = time.time()
        print("[End] Feature preparation:", t4_end)
        print("[Time] Feature preparation:", t4_end - t4_start)
        print("=============================================================")
        
        # 5. model prediction
        t5_start = time.time()
        print("[Start] Model prediction:", t5_start)
        # define result format
        try:
            prediction_scores = [float(score) for score in (xg_reg.predict(content_features.drop(['contentId','origin'], axis = 1)))]
            
            print("len prediction_scores:",len(prediction_scores))

            # 6. construct result schemas
            result = convert_lists_to_dict(string_content_id_list = list(content_features['contentId']), prediction_scores = prediction_scores)
            result = pd.DataFrame(list(result.items()),columns=['contentId', 'score'])
            result = result.merge(content_features[['contentId','origin']], on = 'contentId', how = 'inner')
            result['time_decay'] = 1/((datetime.utcnow() - result['origin']).dt.total_seconds()/3600)
            result['score'] = result['score']*result['time_decay']
            result = result.sort_values(by='score', ascending=False)
            result = convert_lists_to_dict(string_content_id_list = list(result['contentId'].astype(str)), prediction_scores = list(result['score']))
    

            print("len result:",len(result))
        
            response = {
            'statusCode': 200,
            'result': result
            }

            t5_1_end = time.time()
            print("[End] Model prediction:", t5_1_end)
            print("[Time] Model prediction, can predict:", t5_1_end - t5_start)
            print("=============================================================")

        except:
            response = {
            'statusCode': 200,
            'result':'No_match_content_found'
            }

            t5_2_end = time.time()
            print("[End] Model prediction:", t5_2_end)
            print("[Time] Model prediction, cannot predict", t5_2_end - t5_start)
            print("=============================================================")
    
    return response
