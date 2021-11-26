import json
from mongo_client import ping_mongodb, mongo_client

def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    # print(json.dumps(event['body'], indent=4))

    from modules.personalized_content.personalize_content_predictor import personalized_content_predict_main

    response = personalized_content_predict_main(event,
                                      mongo_client,
                                      src_database_name = 'analytics-db',
                                      src_collection_name = 'mlArtifacts',
                                      analytics_db = 'analytics-db',
                                      app_db = 'app-db',
                                      account_collection = 'accounts',
                                      ml_arifact_country_collection = 'mlArtifacts_country',
                                      creator_stats_collection = 'creatorStats',
                                      content_stats_collection = 'contentStats',
                                      model_name = 'xgboost')

    print('prediction of content id: ', event.get('accountId', None),' completed')

    return response

    
