import os
import io
from predict import run
import cv2
import boto3
import numpy as np

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")


def handler(event, context):
    print(event)

    record = event["Records"][0]

    s3bucket = record["s3"]["bucket"]["name"]
    s3object = record["s3"]["object"]["key"]

    s3 = boto3.resource("s3")

    image = s3.Object(s3bucket, s3object).get().get("Body").read()

    image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)
    print(type(image))
    results = run(image)

    _, buffer = cv2.imencode(".jpg", results)
    io_buffer = io.BytesIO(buffer)

    out = s3.Object(OUTPUT_BUCKET, s3object)
    out.upload_fileobj(io_buffer)
