def friend_to_follow(client,
                     selectUser = 'userId',
                     relationships = 'relationships'):
    
    import pymongo # connect to MongoDB
    from pymongo import MongoClient # client connection to MongoDB
    import sklearn
    import pandas as pd
    import json
    import xgboost as xgb
    import bson.objectid
    import pickle
    from datetime import datetime
    from pprint import pprint
    import numpy as np
    from bson.objectid import ObjectId
    import networkx as nx 
 
    appDb = client['app-db']
    analyticsDb = client['analytics-db']
    
    relationship = appDb[relationships]
    relationship = pd.DataFrame(list(relationship.find()))
    
    G = nx.from_pandas_edgelist(relationship,source = 'user', target = 'followedUser')
    def neighborhood(G, node, n):
        path_lengths = nx.single_source_dijkstra_path_length(G, node)
        return [node for node, length in path_lengths.items()
                        if length == n]
    second_degree_frds = pd.DataFrame(neighborhood(G, selectUser, 2),columns=['userId'])
    
    transactionEngagementsCursor = [
        {
            # join to map account id
            '$lookup': {
            'from': 'users', 
            'localField': 'user', 
            'foreignField': '_id', 
            'as': 'users'
            }
        }, {
            # deconstruct users
            '$unwind': {
                'path': '$users'
            }
        }, {
            # summarize by pairing of user ID & content ID
            '$group': {
                '_id': {
                    'accountId': '$users.ownerAccount', # change from user id
                    'contentId': '$targetRef.$id'
                },
                'engangements': {
                    '$push': '$type'
                }
            }
        }, {
            # deconstruct for ease of adding fields
            '$unwind': {
                'path': '$engangements'
            }
        }, {
            # add fields by matching engagement types
            '$addFields': {
                'like': {
                    '$toInt': {
                        '$eq': [
                            '$engangements', 'like'
                        ]
                    }
                },
                'comment': {
                    '$toInt': {
                        '$eq': [
                            '$engangements', 'comment'
                        ]
                    }
                },
                'recast': {
                    '$toInt': {
                        '$eq': [
                            '$engangements', 'recast'
                        ]
                    }
                },
                'quote': {
                    '$toInt': {
                        '$eq': [
                            '$engangements', 'quote'
                        ]
                    }
                }
            }
        }, {
            # summarize to merge all added engagement types
            '$group': {
                '_id': '$_id',
                'like': {
                    '$first': '$like'
                },
                'comment': {
                    '$first': '$comment'
                },
                'recast': {
                    '$first': '$recast'
                },
                'quote': {
                    '$first': '$quote'
                }
            }
        }, {
            # map output format as followed requirement
            '$project': {
                '_id': 0,
                'userId': '$_id.accountId',
                'contentId': '$_id.contentId',
                'like': '$like',
                'comment': '$comment',
                'recast': '$recast',
                'quote': '$quote',
                # alias 'label'
                'engagements': {
                    '$sum': [
                        '$like', 
                        '$comment',
                        '$recast',
                        '$quote'
                    ]
                }
            }
        }
    ]

    # assign result to dataframe
    transaction_engagements = pd.DataFrame(list(client['app-db']['engagements'].aggregate(transactionEngagementsCursor)))
    transaction_engagements = pd.DataFrame(transaction_engagements.groupby('userId')['engagements'].agg('sum')).reset_index()
    
    # Add data temporary
    data = {'userId': [ObjectId('6170067a51db852fb36d2109'),ObjectId('61700a6151db852fd36d2142'),
        ObjectId('617b73f320c181fde77079ca'),ObjectId('6170eb21e5ddcb429e04e7d7'),
       ObjectId('6188a9be31a623e58b6cdd4d'),ObjectId('61a035f03fca318d7220fcdf'),
       ObjectId('618896db31a62333a06cdc13'),ObjectId('6188931d31a6238ab56cdbca'),
       ObjectId('61aedde817602081f094dff4'),ObjectId('61b697834f085629dd2d3031'),
       ObjectId('61b18bfd29800e0ae5e44545'),ObjectId('61c46bf8b723f6566df5d83b'),
       ObjectId('61713ca6d8b0356627fe4f5f'),ObjectId('6170067a51db852fb36d2109'),
       ObjectId('6176d66719931a76d848c5a1'),ObjectId('61710e033e00f69cbfabd308'),
       ObjectId('61763d8db1bf1d7cef72d428'),ObjectId('6170eb21e5ddcb429e04e7d7'),
       ObjectId('6176d62d19931a40ea48c597'),ObjectId('61700a6641fd303d1b0958a5'),
       ObjectId('6176d64119931a7a0f48c59c'),ObjectId('618896db31a62333a06cdc13')],
        'engagements': [12,15,14,23,45,17,16,20,34,58,1,
                        7,7,9,52,14,87,35,12,41,21,11]
        }
    add_data = pd.DataFrame(data).drop_duplicates(subset = ['userId'])
    transaction_engagements = (transaction_engagements.append(add_data)).reset_index(drop = True)
    
    sorted_second_degree_frd = second_degree_frds.merge(transaction_engagements, on = 'userId', how = 'left' ).sort_values(by='engagements', ascending=False)
    
    if len(sorted_second_degree_frd) < 14:
        add_sorted_second_degree_frd = (sorted_second_degree_frd.append(transaction_engagements.sort_values(by='engagements', ascending=False)[:14])).drop_duplicates(subset = ['userId'], keep= 'first').reset_index(drop = True)
        add_sorted_second_degree_frd['index'] = np.arange(len(add_sorted_second_degree_frd))
        add_sorted_second_degree_frd = add_sorted_second_degree_frd.to_dict(orient='records')
    else : 
        add_sorted_second_degree_frd = sorted_second_degree_frd
        add_sorted_second_degree_frd['index'] = np.arange(len(add_sorted_second_degree_frd))
        add_sorted_second_degree_frd = add_sorted_second_degree_frd.to_dict(orient='records')
    
    print("len result:",len(add_sorted_second_degree_frd ))
    print(add_sorted_second_degree_frd)
    
    response = {
        'statusCode': 200,
        'result': add_sorted_second_degree_frd
    }

    return response
    

def friend_to_follow_main(client):
    
    friend_to_follow(client,
                     selectUser = 'userId',
                     relationships = 'relationships') 
    

    
    return