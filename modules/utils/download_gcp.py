def download_gcp_main(local_path):
    import boto3
    import botocore

    BUCKET_NAME = 'ml-dev.castcle.com' # replace with your bucket name
    KEY = 'gcp_data-science_service-account_key.json' # replace with your object key

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, local_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    return