"""
AWS S3 클라이언트
Spring Boot와 동일한 S3 버킷에 접근합니다.
"""
import boto3
from botocore.exceptions import ClientError
from loguru import logger

from app.core.config import settings


class S3Client:
    """S3 파일 접근 클라이언트"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of S3 client"""
        if self._client is None:
            self._client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
        return self._client

    def get_image(self, s3_key: str) -> bytes:
        """
        S3에서 이미지 다운로드

        Args:
            s3_key: S3 객체 키 (예: contracts/2024/01/image.jpg)

        Returns:
            이미지 바이트 데이터

        Raises:
            FileNotFoundError: S3 객체가 없을 때
            RuntimeError: S3 접근 오류
        """
        try:
            response = self.client.get_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=s3_key
            )
            image_bytes = response['Body'].read()
            logger.debug(f"S3에서 이미지 다운로드 완료: {s3_key}")
            return image_bytes

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"S3 객체를 찾을 수 없음: {s3_key}")
                raise FileNotFoundError(f"S3 객체를 찾을 수 없음: {s3_key}")
            else:
                logger.error(f"S3 접근 오류: {e}")
                raise RuntimeError(f"S3 접근 오류: {e}")

    def upload_file(self, file_bytes: bytes, s3_key: str, content_type: str = "image/jpeg") -> str:
        """
        S3에 파일 업로드

        Args:
            file_bytes: 파일 바이트 데이터
            s3_key: S3 객체 키
            content_type: MIME 타입

        Returns:
            S3 URL
        """
        try:
            self.client.put_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type
            )
            url = f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            logger.info(f"S3 업로드 완료: {s3_key}")
            return url

        except ClientError as e:
            logger.error(f"S3 업로드 오류: {e}")
            raise RuntimeError(f"S3 업로드 오류: {e}")

    def delete_file(self, s3_key: str) -> bool:
        """
        S3 파일 삭제

        Args:
            s3_key: S3 객체 키

        Returns:
            성공 여부
        """
        try:
            self.client.delete_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=s3_key
            )
            logger.info(f"S3 파일 삭제 완료: {s3_key}")
            return True

        except ClientError as e:
            logger.error(f"S3 삭제 오류: {e}")
            return False

    def file_exists(self, s3_key: str) -> bool:
        """
        S3 파일 존재 여부 확인

        Args:
            s3_key: S3 객체 키

        Returns:
            존재 여부
        """
        try:
            self.client.head_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=s3_key
            )
            return True
        except ClientError:
            return False
