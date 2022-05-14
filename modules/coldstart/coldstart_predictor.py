
# main function of personalize content model prediction
# 1. get input
# 2. loading model
# 3. feature preparation
# 4. model prediction
# 5. construct result schemas

import pandas as pd
#----------------------------------------------------------------------------------------------------------------
#! Fixme
def retrive_junk_score(testcase):
    """
    query content junk score from testcase
    """
    from mongo_client import mongo_client as client
    mycol_contentfiltering = client['analytics-db']['contentfiltering'] #! Fixme
    # retrive content in contentfiltering
    query_data_content = list(mycol_contentfiltering.aggregate([
                                         {'$match': {'contentId': {'$in':testcase}}}
                                                             ,{'$project': {'_id' : 1 ,'contentId':1,'junkOutput':1,'textDiversity':1}}]))
    print('all = ', len(testcase),'-> havescore = ', len(query_data_content))
    return query_data_content

def query_content_junkscore(test_case):
    """
    Main junk feature
    Retrive junkscore and then recalulate if can't find
    """
    #! Fixme  
    def extract_score(x):
        #2 62454ea1becd94109a8a229d {'class': 'junk', 'score': 0.0}
        if isinstance(x, dict):
            return x['score']
        else:
            print("cant find x['score']")
            return 0.5

    import time
    import pandas as pd
    from bson.objectid import ObjectId
    # find junk score
    start_time = time.time()
    query_data_content = retrive_junk_score(test_case)

    # create dummy table
    junkcol_name = ['_id', 'contentId', 'junkOutput', 'textDiversity']
    junkcol_df = pd.DataFrame({},columns = junkcol_name)

    # merge result with dummy table
    contentfiltering_df = pd.concat([junkcol_df, pd.DataFrame(query_data_content)])
    contentfiltering_df = contentfiltering_df.drop('_id', axis=1)
    contentfiltering_df = contentfiltering_df.rename({'contentId':'content_id','junkOutput':'junkscore'},axis = 1)
    print('contentfiltering_df', contentfiltering_df.head())
    contentfiltering_df['junkscore'] = contentfiltering_df['junkscore'].fillna(0).apply(extract_score) # Call extract_score(.apply function)
    contentfiltering_df['textDiversity'] = contentfiltering_df['textDiversity'].fillna(0.5) #add diversity
    print(" retrieve junk score --- %s seconds ---" % (time.time() - start_time))
    
    #find not have score
    contentid_no_junk_score = list(set(test_case) - set(contentfiltering_df['content_id'].tolist()))
    result = contentfiltering_df.copy() #return contentfiltering_df if all contents have junk score
    print('len no junk = ',len(contentid_no_junk_score))

    # junk score = 0 if not have
    #find not have score
    contentid_no_junk_score = list(set(test_case) - set(contentfiltering_df['content_id'].tolist()))
    result = contentfiltering_df.copy() #return contentfiltering_df if all contents have junk score
    print('len no junk = ',len(contentid_no_junk_score))

    df_null = pd.DataFrame(contentid_no_junk_score).rename({0:'content_id'},axis = 1)
    result = pd.concat([result, df_null])
    result['junkscore'] = result['junkscore'].fillna(0.5)
    result['textDiversity'] = result['textDiversity'].fillna(0.5)
    
    print(" recalulate -> df_junkscore --- %s seconds ---" % (time.time() - start_time))
    return result
#----------------------------------------------------------------------------------------------------------------

def _add_fields(
    mongo_client, 
    result_df: pd.DataFrame) -> pd.DataFrame:
    '''
    @title add fields to result (app-db.guestfeeditems)
        Add authorId in collection guestfeeditems
        Add originalContent = originalPost._id
        Add originalAuthor = originalPost.author.id
    '''
    import numpy as np

    # Add authorId -> author
    _project = {
        'contentId': 1, 
        'authorId': 1
    }

    _get_authorId_df = pd.DataFrame(list(
        mongo_client['analytics-db']['contentStats'].find(
            {}, _project
            )
        )
    )

    _get_authorId_df = _get_authorId_df.drop(['_id'], axis=1, errors='ignore')
    _get_authorId_df = _get_authorId_df.rename(columns={'contentId': 'content'})

    # drop authorId from the result_df before join
    result_df = result_df.drop(['authorId', 'author'], axis=1, errors='ignore')

    result_df = pd.merge(result_df, _get_authorId_df, how='left', on='content')
    result_df = result_df.rename(columns={'authorId': 'author'})

    # Add originalPost
    _aggreate = [
        {
            '$match': {
                'originalPost': {
                    '$exists': True
                }
            }
        }, {
            '$project': {
                'content': '$_id', 
                'originalContent': '$originalPost._id', 
                'originalAuthor': '$originalPost.author.id'
            }
        }
    ]
    _get_original_post_df = pd.DataFrame(list(
        mongo_client['app-db']['contents'].aggregate(_aggreate)
        ))
    _get_original_post_df = _get_original_post_df.drop(['_id'], axis=1, errors='ignore')
    # drop authorId from the result_df before join
    result_df = result_df.drop(['originalContent', 'originalAuthor'], axis=1, errors='ignore')
    result_df = pd.merge(result_df, _get_original_post_df, how='left', on='content')

    # clean NaN -> None
    result_df = result_df.replace({np.nan: None})

    return result_df

def cold_start_by_counytry_scroing( mongo_client,
                                    updatedAtThreshold,
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'guestfeeditems',
                                    saved_data_temp = 'guestfeeditemstemps',
                                    model_name = 'xgboost'
                                    ):
    

    import pickle
    from datetime import datetime
    from pprint import pprint
    from datetime import datetime, timedelta
    import pymongo

    # connect to database
    appDb = mongo_client['app-db']
    analyticsDb = mongo_client['analytics-db']

    # define feature preparation function from content id list
    def prepare_features(mongo_client, 
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str,
                     updatedAtThreshold = updatedAtThreshold):
        
        from mongo_client import mongo_client as client #! Fixme
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
                    'ageScore': '$aggregator.ageScore',
                    'updatedAt': 1
                }
            }
        ]
        # assign result to dataframe
        # alias 'contentFeatures'

        #content_features = pd.DataFrame(list(mongo_client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
 
        # create dummy table
        features_name = ['updatedAt', 'contentId', 'likeCount', 'photoCount', 'characterLength', 'ageScore', 'commentCount', 'recastCount', 'quoteCount', 'creatorContentCount', 'creatorLikedCount', 'creatorCommentedCount', 'creatorRecastedCount', 'creatorQuotedCount']
        content_features = pd.DataFrame({},columns = features_name)

        # query feature
        mycol_contentStats = mongo_client['analytics-db']['contentStats']
        query_contentStats = list(mycol_contentStats.aggregate(contentFeaturesCursor))

        # merge result with dummy table
        query_contentStats_df = pd.concat([content_features, pd.DataFrame(query_contentStats).rename({'_id':'contentId'},axis = 1)]).fillna(0) # null -> 0
        #print('query_contentStats_df', query_contentStats_df.head())
 
        return query_contentStats_df

    contentFeatures = prepare_features(mongo_client = mongo_client, # default
                                        analytics_db = 'analytics-db',
                                        content_stats_collection = 'contentStats',
                                        creator_stats_collection = 'creatorStats',
                                        updatedAtThreshold = updatedAtThreshold).rename({'updatedAt':'origin'},axis = 1)
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
        
        contentFeatures_for_scoring = contentFeatures.drop(['origin'], axis=1)
        
        # scoring process
        contentFeatures_for_pred = contentFeatures_for_scoring.drop(['contentId'], axis = 1)
        model_predict_content = model_load.predict(contentFeatures_for_pred)
        score = pd.DataFrame(model_predict_content, columns = ['score'])
        
        # set up schema
        content_list = contentFeatures[['contentId']].reset_index(drop = True)
        content_score = pd.concat([content_list,score],axis =1)
        content_score['countryCode'] = countryId
        content_score['type'] = "content"
        content_score['updatedAt'] = datetime.utcnow() 
        content_score['createdAt'] = datetime.utcnow() 
        content_score = content_score.rename({"contentId":"content"},axis = 1)
        
        # add decay function 
        content_score_add_decay_function = content_score.merge(contentFeatures[['contentId','origin']],right_on = 'contentId', left_on = 'content', how = 'inner')
        list_contentId = content_score_add_decay_function['content'].tolist()
        print('contentId: ', list_contentId)
        
        # Retreive additional score #! Fixme
        junk_score_df = query_content_junkscore(list_contentId).rename(columns={'content_id': 'content'})  #! Fixme
        content_score_add_decay_function = content_score_add_decay_function.merge(junk_score_df, on = 'content', how = 'left')
        content_score_add_decay_function['junkscore'] = content_score_add_decay_function['junkscore'] + 0.01 #[0.0-1.0] ->[0.01-1.01]
        content_score_add_decay_function['textDiversity'] = content_score_add_decay_function['textDiversity'] + 0.01 #[0.0-1.0] ->[0.01-1.01]
        print('result_junk: ', content_score_add_decay_function['junkscore'].tolist())
        print('textDiversity: ', content_score_add_decay_function['textDiversity'].tolist())
        #print('result_column', content_score_add_decay_function.columns.tolist())
            
        # Personalize scoring
        content_score_add_decay_function['time_decay'] = 1/((content_score_add_decay_function['createdAt']-content_score_add_decay_function['origin']).dt.total_seconds()/3600)
        content_score_add_decay_function['score'] = content_score_add_decay_function['score']*content_score_add_decay_function['time_decay']*content_score_add_decay_function['junkscore']*content_score_add_decay_function['textDiversity'] #! Fixme
        content_score = content_score_add_decay_function[['content','score','countryCode','type','updatedAt','createdAt']]
        print('result: ', content_score_add_decay_function)
        
        #set limit
        content_score = content_score.sort_values(by='score', ascending=False)
        print("content_score.shape1: " , content_score.shape)
        content_score = content_score.iloc[:2000,]
        print("content_score.shape2: " , content_score.shape)
        
        # append result
        result = result.append(content_score)
        print("len result: " , len(result))

        # join authorId in result
        result = _add_fields(mongo_client=mongo_client, result_df=result)
        
     # update collection
    result.reset_index(inplace=False)
    
    data_dict = result.to_dict("records")
    
    # save to temporary storage
    saved_data_country_temporary.insert_many(data_dict)
    print('done_save')

    # save to target stoage
    saved_data_country_temporary.rename(saved_data, dropTarget = True)
    print('done_move')
    
    saved_data_country.create_index([("countryCode", pymongo.DESCENDING)])
    
def coldstart_score_main(
        mongo_client,
        updatedAtThreshold) -> None:
    
    cold_start_by_counytry_scroing( mongo_client,
                                    updatedAtThreshold = updatedAtThreshold,
                                    saved_model = 'mlArtifacts_country',
                                    saved_data = 'guestfeeditems',
                                    saved_data_temp = 'guestfeeditemstemp',
                                    model_name = 'xgboost')
    return
