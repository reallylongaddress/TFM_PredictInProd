# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
LOCAL_PATH=./data/train_1k.csv

# project id
PROJECT_ID=lw-project-351306

# bucket name
BUCKET_NAME=wagon-data-893-dimarco2

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=data

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=europe-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

# BUCKET_NAME=XXX

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

# REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.2

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=TaxiFareModel
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')


train_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc


##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn api.fast:app --port=8003 --reload  # load web server with code autoreload

### DBD Added

DOCKER_IMAGE_NAME=dbd_lw_docker_image_20220815_8

docker_build_local:
	docker build -t ${DOCKER_IMAGE_NAME} .

docker_run_local:
	docker run -e PORT=8080 -p 8000:8080 ${DOCKER_IMAGE_NAME}

docker_run_local_interactive:
	docker run -it -e PORT=8080 -p 8000:8080 ${DOCKER_IMAGE_NAME} sh

#build docker image locally
docker_build_gcr_local:
	docker buildx build --platform linux/amd64 -t  eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

#build docker image on GCP
docker_build_gcr_cloud:
	gcloud builds submit -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_push_gcr:
	docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_run_gcr_local:
	docker run -e PORT=8080 -p 8000:8080 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

gcloud_deploy_gcp:
	gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1

#gcloud_shutdown_gcp
