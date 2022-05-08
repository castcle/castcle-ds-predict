from mongo_client import mongo_client as client 

def query_contentinfo_db(content_id, client):
    '''
    select * from mydb.mycol_contents where contentId = 'content_id'
    '''
    from bson.objectid import ObjectId
    mydb  = client['app-db']
    mycol_contentinfo = mydb['contentinfo']
    myquery = {'contentId':ObjectId(content_id)}
    contentinfo_query = list(mycol_contentinfo.find(myquery))
    return contentinfo_query

def clean_text(txt_):
    """
    remove url, hashtag, newline. emoji, doublespace
    """
    import re
    detect_url = '((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    txt_ = re.sub(detect_url, '', txt_)
    detect_hashtag = r"#(\w+)"
    txt_ = re.sub(detect_hashtag, '', txt_)
    detect_newline = r'(\n+)'
    txt_ = re.sub(detect_newline, '', txt_)
    detect_emoji = r'\d+(.*?)(?:\u263a|\U0001f645)'
    txt_ = re.sub(detect_emoji, '', txt_)
    detect_emoji2 = r"(u00a9u00ae[u2000-u3300]ud83c[ud000-udfff]ud83d[ud000-udfff]ud83e[ud000-udfff])"
    txt_ = re.sub(detect_emoji2, "", txt_);
    detect_doublespace = r' +'
    txt_ = re.sub(detect_doublespace, ' ', txt_).strip()
    detect_doublespace = r'[^a-zA-Z]+'
    txt_ = re.sub(detect_doublespace, ' ', txt_).strip()
    return txt_


def sentence_encoding(message_):
    """
    Call sentence encoding model
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(message_, convert_to_tensor=True)
    return embeddings

# Retreive content not in contentfiltering
def query_1min_content(duration):
    """
    Retreive translatedEN from contentfiltering in <duration> miniute and limit 1000 sentent
    """
    mycol_contentfiltering = client['analytics-db']['contentfiltering']
    query_data_content = list(mycol_contentfiltering.aggregate([
                                          { '$match':{'$expr': {'$gt': ["$timestamp",
                    { '$dateSubtract': { 'startDate': "$$NOW", 'unit': "minute", 'amount': duration }}]}}}
                                                           ,{'$project': {'_id' : 1 ,'massageInEN':1}}
                                                            ,{'$limit':1000}
 
                                                                ])) 
    return query_data_content

def embeddings_sentence(duration):
    '''
    write sentence_embeddings to mongodb
    '''
    def extract_message(x):
    x = clean_text(x)
    return x

    import pandas as pd
    from datetime import datetime

    # query data
    query_data_content = query_1min_content(duration)

    # if insufficate content(50) duration = 100 min
    if query_data_content != []:
        if len(query_data_content) <= 50:
            # query data
            duration = 100 #100min
            query_data_content = query_1min_content(duration)
        else:
            pass
    else:
        # query data
        duration = 100
        query_data_content = query_1min_content(duration)
    
    print('duration', duration)
    message_content = pd.DataFrame(query_data_content)
    print(message_content.head)
    list_of_message = list(set(message_content['massageInEN'].apply(extract_message).tolist()))
    print('list_of_message', len(list_of_message))

    # sentence_encoding
    sentence_embedd = sentence_encoding(list_of_message).tolist()
    print('compare_embeddings', len(sentence_embedd))

    # write to mongo
    writeCol = client['analytics-db']['sentence_embeddings']
    try: # create if not exists
        # drop collection
        writeCol.drop()
        client['analytics-db'].create_collection('sentence_embeddings')
    except Exception  as e:  # if error 
        print(e)
        pass
    try: # try to write
        write_mongo_df = {}
        write_mongo_df['sentence_embeddings'] = sentence_embedd
        write_mongo_df['updatedAt'] = datetime.utcnow()
        writeCol.insert_one(write_mongo_df)
        print('write done')
    except Exception  as e:  # if error  
        print(e)
        pass
