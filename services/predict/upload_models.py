import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_models_to_s3():
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'eu-north-1')
    )
    
    bucket_name = os.getenv('AWS_BUCKET_NAME')
    
    # List of models to upload
    models = {
        'model_checkpoint.pth': 'model_checkpoint.pth',  # local file : s3 key
        'best_model.pth': 'best_model.pth'
    }
    
    # Create bucket if it doesn't exist
    try:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': os.getenv('AWS_REGION', 'eu-north-1')}
        )
        print(f"Created bucket: {bucket_name}")
    except s3_client.exceptions.BucketAlreadyExists:
        print(f"Bucket {bucket_name} already exists")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket {bucket_name} already owned by you")
    
    # Upload each model
    for local_file, s3_key in models.items():
        try:
            print(f"Uploading {local_file} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_file, bucket_name, s3_key)
            print(f"Successfully uploaded {local_file}")
        except FileNotFoundError:
            print(f"Error: Could not find {local_file}")
        except Exception as e:
            print(f"Error uploading {local_file}: {str(e)}")

if __name__ == "__main__":
    upload_models_to_s3()