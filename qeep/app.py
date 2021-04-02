import os
import io
from predict import run
import cv2
import boto3

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")


def handler(event, context):
    print(event)
    print(event["body"])

    record = event["Records"][0]

    s3bucket = record["s3"]["bucket"]["name"]
    s3object = record["s3"]["object"]["key"]

    s3Path = "s3://" + s3bucket + "/" + s3object

    image = cv2.imread(s3Path)
    results = run(image)

    _, buffer = cv2.imencode(".jpg", results)
    io_buffer = io.BytesIO(buffer)

    s3client = boto3.client("s3")
    s3client.upload_fileobj(io_buffer, OUTPUT_BUCKET, s3object)
