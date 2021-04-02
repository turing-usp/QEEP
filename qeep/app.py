import boto3
from predict import run
import cv2

def lambda_handler(event, context):
    print(event)
    sender, message, time = event["sender"], event["message"], event["time"]
    handle_response(sender, message, time)

if __name__ == "__main__":
    result = run("poke.jpg",min_conf=-0.01, visualize=-1)
    cv2.imshow("Predicoes", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
