"""
   api front-lambda interface
"""
import os
import io
from hashlib import md5
import boto3
import filetype

INPUT_BUCKET = os.getenv("INPUT_BUCKET")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")


def file_extension(file) -> str:
    """
    Checks for the extension of passed file and
    returns it as string
    """

    kind = filetype.guess(file)
    if kind is None:
        return ""

    return "." + kind.extension


def lambda_handler(event, _context):
    """
    Lambda event handler
    """

    s3_resource = boto3.resource("s3")

    print(event)

    buffer = event["body"]
    io_buffer = io.BytesIO(buffer)

    filename = md5(buffer) + file_extension(buffer)
    location = boto3.client("s3").get_bucket_location(Bucket=INPUT_BUCKET)[
        "LocationConstraint"
    ]

    bucket = s3_resource.Object(INPUT_BUCKET, filename)
    bucket.upload_fileobj(io_buffer)

    outputlink = (
        f"https://{OUTPUT_BUCKET}.s3-{location}.amazonaws.com/{filename}"
    )

    return {"statusCode": 202, "outputLink": outputlink}
