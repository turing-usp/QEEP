FROM public.ecr.aws/lambda/python:3.8

WORKDIR /var/task

RUN ["yum", "update", "-y"]
RUN ["yum", "install", "-y", "opencv-python"]

COPY requirements.txt .

RUN ["pip", "install", "-r", "requirements.txt", "--no-cache-dir"]

COPY . .

CMD ["qeep_app.handler"]
