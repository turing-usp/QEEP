#!/bin/bash
DEPLOY_URL=154769901104.dkr.ecr.sa-east-1.amazonaws.com
REPO_NAME_API=turing/qeep_api
API_FUNCTION=turing-qeep-api

# update api
sudo docker build -f Dockerfile.api . -t $REPO_NAME_API
aws ecr get-login-password --region sa-east-1 | sudo docker login --username AWS --password-stdin $DEPLOY_URL
sudo docker tag $REPO_NAME_API:latest $DEPLOY_URL/$REPO_NAME_API:latest
sudo docker push $DEPLOY_URL/$REPO_NAME_API:latest
aws lambda update-function-code --function-name $API_FUNCTION --image-uri $DEPLOY_URL/$REPO_NAME_API:latest
