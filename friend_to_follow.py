import json
from mongo_client import ping_mongodb, mongo_client
import datetime


def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    print(json.dumps(event, indent=4))
    
    from modules.friend_to_follow.friend_to_follow \
        import friend_to_follow_main

    # call modules main function    
    friend_to_follow_main(event, client=mongo_client)
    
    print('friend_to_follow: ',' completed')

    # return output as status code and timestamp
    return {
        "statusCode": 200,
        "predicted_at": str(datetime.datetime.now())
    }
    
