import boto3
import os
import io
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')
        logger.info(f"The bucket name is {self.bucket_name}")

    def download_model(self, model_key):
        """Download model from S3 and return as bytes"""
        try:
            logger.info(f"Downloading {model_key} from bucket {self.bucket_name}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=model_key)
            model_bytes = response['Body'].read()
            logger.info(f"Successfully downloaded {model_key}, size: {len(model_bytes)} bytes")
            return io.BytesIO(model_bytes)
        except ClientError as e:
            logger.error(f"Error downloading model from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise