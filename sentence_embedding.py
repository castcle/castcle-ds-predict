import json
from mongo_client import ping_mongodb
import datetime

def download_model():
    BUCKET_NAME = 'ml-dev.castcle.com'
    file_path = 'sentence_transformer/SentenceTransformer.pkl'
    file_name = 'SentenceTransformer.pkl'
    
    s3_client = boto3.client('s3')
    s3_client.download_file(BUCKET_NAME, file_path, file_name)
    print('Sucessfull_download:',file_name)
    
    from os import listdir
    onlyfiles = [f for f in listdir('./')]
    print('file_list :', onlyfiles)

# download SentenceTransformer.pkl model
download_model()

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
