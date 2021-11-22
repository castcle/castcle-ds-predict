import json
from mongo_client import ping_mongodb, mongo_client


# Example event schema
# event = {
#     'accountId': '6170063351db852b0e6d20fc',
#     'contents': [
#         '6188a77b1ec609099d728303',
#         '6188ed841ec60953167283f4'
#     ]
# }


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
                                      creator_stats_collection = 'creatorStats',
                                      content_stats_collection = 'contentStats',
                                      dst_database_name = 'analytics-db',
                                      dst_collection_name = 'feedItems_test',
                                      model_name = 'xgboost')

    print('prediction of content id: ', event.get('userId', None),' completed')

    # return {
    #     'statusCode': 200,
    #     'body': json.dumps({
    #         'msg': 'Hi, there!'
    #     })
    # }

    return response # change together with new result schema 2021-11-17 by coupon
