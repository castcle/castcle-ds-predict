import json
from mongo_client import ping_mongodb, mongo_client


def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    print(json.dumps(event['body'], indent=4))

    return {
        'statusCode': 200,
        'body': json.dumps({
            'msg': 'Hi, there!'
        })
    }
