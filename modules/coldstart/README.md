# Coldstart predictor
content recommendation (prediction part) for anonymous or non-active user by country
## scenario
![coldstart-predictor](https://user-images.githubusercontent.com/91544452/146479705-e8a43619-41c7-458e-9639-32d17eb49377.JPG)

run this predictor every 30 minute (subject to change) then save result to mongodb
1. prepare aggregated features from "contentStats" & "creatorStats"
2. retrieve model artifact by country 
3. save prediction result to mongodb

## Prepare features
feature preparation from "contentStats" & "creatorStats" for ultilize as feature in model prediction
```python
    def prepare_features(client, 
                     analytics_db: str,
                     content_stats_collection: str,
                     creator_stats_collection: str):

        '''
        feature preparation from "contentStats" & "creatorStats" for ultilize as feature in model prediction
        '''
    
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

                }
            }
        ]


        content_features = pd.DataFrame(list(client[analytics_db][content_stats_collection].aggregate(contentFeaturesCursor))).rename({'_id':'contentId'},axis = 1)
    
        return content_features

    contentFeatures = prepare_features(client = client, # default
                                        analytics_db = 'analytics-db',
                                        content_stats_collection = 'contentStats',
                                        creator_stats_collection = 'creatorStats')

    mlArtifacts_country = analyticsDb[saved_model]
    ml_set = pd.DataFrame(list(mlArtifacts_country.find()))
    
    saved_data_country = appDb[saved_data]
    
    saved_data_country_temp = appDb[saved_data_temp]
    

    saved_data_country_temp.drop({})
```
## Retrieve model artifact
retrieve model artifact of the user in case of present or country model artifact in case of model artifact absent
```python
def load_model_from_mongodb(collection, model_name, account):

	'''
	retrieve model artifact of the user in case of present or country model artifact in case of model artifact absent
	'''    

	json_data = {}
	data = collection.find({
		'account': account,
		'model': model_name
	})

	for i in data:
		json_data = i

	pickled_model = json_data['artifact']

	return pickle.loads(pickled_model)
```
## Making coldstart prediction
After model retrieved, make predictions. Then, save the result to mongodb
```python
result = pd.DataFrame()
for countryId in list(ml_set.account.unique()):
	pprint(countryId)
	xg_reg_load = load_model_from_mongodb(collection=mlArtifacts_country,
									account= countryId,
									model_name= model_name)

	content_test = contentFeatures

	a = pd.DataFrame(xg_reg_load.predict(content_test.drop(['contentId'], axis = 1)), columns = ['score'])
	b = contentFeatures[['contentId']].reset_index(drop = True)
	c = pd.concat([b,a],axis =1)
	c['countryCode'] = countryId
	c['type'] = "content"
	c['updatedAt'] = datetime.utcnow() 
	c['createdAt'] = datetime.utcnow() 
	c = c.rename({"contentId":"content"},axis = 1)
	c = c.sort_values(by='score', ascending=False)
	c = c.iloc[:2000,]
	result = result.append(c)  

	
# update collection
result.reset_index(inplace=False)

data_dict = result.to_dict("records")

saved_data_country_temp.insert_many(data_dict)
print('done_save')


saved_data_country_temp.rename(saved_data, dropTarget = True)
print('done_move')
```
