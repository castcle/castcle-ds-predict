import json
from mongo_client import ping_mongodb, mongo_client


def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    print(json.dumps(event, indent=4))
    
    from modules.coldstart_prediction.coldstart_predictor \
        import coldstart_predictor_main
        
    coldstart_predictor_main(client=mongo_client)
    
    print('prediction of coldstart: ',' completed')

    return {
        "statusCode": 200
    }