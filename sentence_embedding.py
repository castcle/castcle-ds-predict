import json
from mongo_client import ping_mongodb
import datetime


def handle(event, context):
    if event.get("source") == "serverless-plugin-warmup":
        ping_mongodb()
        print("WarmUp - Lambda is warm!")
        return

    print(json.dumps(event, indent=4))
    
    from modules.embeddings.sentence_embeddings import embeddings_sentence

    # call modules main function  
    embeddings_sentence(duration = 1)
    
    print('embeddings_sentents: ',' completed')

    # return output as status code and timestamp
    return {
        "statusCode": 200,
        "predicted_at": str(datetime.datetime.now())
    }
