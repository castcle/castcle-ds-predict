import json
from mongo_client import ping_mongodb, mongo_client


#! simulate mocked event
## following creator's content
'''
{
    "userId": "6170067a51db852fb36d2109",
    "contents": [
        "6188a77b1ec609099d728303",
        "6188ed841ec60953167283f4"
    ]
}
'''
def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    # print(json.dumps(event['body'], indent=4))

    from modules.personalized_content.personalize_content_predictor import personalized_content_predict_main

    personalized_content_predict_main(event,
                                      mongo_client,
                                      src_database_name = 'analytics-db',
                                      src_collection_name = 'mlArtifacts',
                                      analytics_db = 'analytics-db',
                                      creator_stats_collection = 'creatorStats',
                                      content_stats_collection = 'contentStats',
                                      dst_database_name = 'analytics-db',
                                      dst_collection_name = 'feedItems_test',
                                      model_name = 'xgboost')

    print('prediction of content id: ', event.get('userId', None),' completed')

    return {
        'statusCode': 200,
        'body': json.dumps({
            'msg': 'Hi, there!'
        })
    }
