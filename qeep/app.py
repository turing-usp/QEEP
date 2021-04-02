import boto3


def lambda_handler(event, context):
    print(event)
    sender, message, time = event["sender"], event["message"], event["time"]
    handle_response(sender, message, time)
