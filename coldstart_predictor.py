import json
from mongo_client import ping_mongodb, mongo_client
import datetime


def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    print(json.dumps(event, indent=4))
    
    from modules.coldstart.coldstart_predictor \
        import coldstart_score_main

    # call modules main function    
    coldstart_score_main(
        mongo_client=mongo_client,
        updatedAtThreshold=30.0)
    
    print('prediction of coldstart: ',' completed')

    # return output as status code and timestamp
    return {
        "statusCode": 200,
        "predicted_at": str(datetime.datetime.now())
    }