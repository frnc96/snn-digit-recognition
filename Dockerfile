FROM python:3

# set working directory
WORKDIR /code

# copy dependencies file
COPY ./requirements.txt ./

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy code base
COPY ./src /code/src