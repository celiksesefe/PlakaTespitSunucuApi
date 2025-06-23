# app/s3_utils.py

import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
from io import BytesIO
from typing import Optional, Tuple
import uuid
import os
from datetime import datetime
from .config import (
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY, 
    AWS_S3_BUCKET_NAME, 
    AWS_S3_REGION,
    AWS_S3_BASE_URL
)
from .exceptions import APIException

logger = logging.getLogger(__name__)

class S3Manager:
    """AWS S3 operations manager"""
    
    def __init__(self):
        self.bucket_name = AWS_S3_BUCKET_NAME
        self.region = AWS_S3_REGION
        self.base_url = AWS_S3_BASE_URL
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize S3 client with error handling"""
        try:
            if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME, AWS_S3_REGION]):
                raise APIException("AWS S3 credentials not properly configured", 500)
            
            self._client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_S3_REGION
            )
            
            # Test connection by checking if bucket exists
            self._client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise APIException("AWS credentials not configured", 500)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' not found")
                raise APIException(f"S3 bucket '{self.bucket_name}' not found", 500)
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket '{self.bucket_name}'")
                raise APIException(f"Access denied to S3 bucket", 500)
            else:
                logger.error(f"S3 initialization error: {e}")
                raise APIException(f"S3 initialization failed: {str(e)}", 500)
        except Exception as e:
            logger.error(f"Unexpected error initializing S3: {e}")
            raise APIException(f"S3 initialization failed: {str(e)}", 500)
    
    def generate_s3_key(self, filename: str) -> str:
        """Generate S3 key with organized folder structure"""
        # Get file extension
        ext = os.path.splitext(filename)[1]
        
        # Create organized path: plates/YYYY/MM/DD/uuid.ext
        now = datetime.utcnow()
        date_path = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        unique_filename = f"{uuid.uuid4()}{ext}"
        
        s3_key = f"plates/{date_path}/{unique_filename}"
        return s3_key
    
    def upload_image(self, image_bytes: bytes, filename: str, content_type: str = None) -> Tuple[str, str]:
        """
        Upload image to S3 and return S3 key and public URL
        
        Args:
            image_bytes: Image file content as bytes
            filename: Original filename (used for extension)
            content_type: MIME type of the image
            
        Returns:
            Tuple of (s3_key, public_url)
        """
        try:
            # Generate S3 key
            s3_key = self.generate_s3_key(filename)
            
            # Determine content type if not provided
            if not content_type:
                ext = os.path.splitext(filename)[1].lower()
                content_type_map = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.bmp': 'image/bmp',
                    '.tiff': 'image/tiff',
                    '.webp': 'image/webp'
                }
                content_type = content_type_map.get(ext, 'image/jpeg')
            
            # Upload to S3
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=BytesIO(image_bytes),
                ContentType=content_type,
                ACL='public-read',  # Make images publicly accessible
                Metadata={
                    'original_filename': filename,
                    'upload_timestamp': datetime.utcnow().isoformat(),
                    'service': 'plaka-tespit-api'
                }
            )
            
            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded image to S3: {s3_key}")
            return s3_key, public_url
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"S3 upload failed with error {error_code}: {e}")
            raise APIException(f"Failed to upload image to S3: {error_code}", 500)
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise APIException(f"Failed to upload image to S3: {str(e)}", 500)
    
    def delete_image(self, s3_key: str) -> bool:
        """
        Delete image from S3
        
        Args:
            s3_key: S3 object key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted image from S3: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete S3 object {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting S3 object {s3_key}: {e}")
            return False
    
    def get_image_url(self, s3_key: str) -> str:
        """Generate public URL for S3 object"""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
    
    def check_image_exists(self, s3_key: str) -> bool:
        """Check if image exists in S3"""
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

# Global S3 manager instance
s3_manager = S3Manager()