from bson.objectid import ObjectId


def convert_objectid(_id) -> ObjectId:
    '''
    Convert _id: string into mongodb ObjectId
    '''
    from bson import ObjectId
    
    if not isinstance(_id, ObjectId):
        _id = ObjectId(_id)
    elif isinstance(_id, ObjectId):
        _id = _id
    
    return _id