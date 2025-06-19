// Defining a Jenkins pipeline
pipeline {
    agent any // Specifies that the pipeline can run on any available Jenkins agent

    environment {
        VENV_DIR = 'venv' // Directory name for the Python virtual environment
        GCP_PROJECT = 'single-arcadia-463020-t4' // GCP project ID where resources will be managed
        GCLOUD_PATH = '/var/jenkins_home/gcloud/google-cloud-sdk/bin' // Path to the Google Cloud SDK
    }

    stages {

        // Stage 1: Cloning the GitHub repository
        stage('Cloning Github repo to Jenkins') {
            steps {
                script{
                echo 'Cloning repository from GitHub to Jenkins workspace' // Log message for cloning step

                // Cloning the repository from GitHub using the specified branch and credentials
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Sarvesh-Yadav-5201/MLOps_Project_1_Hotel_Reservation_prediction.git']])
                }
            }
        }

        // Stage 2: Setting up a Python virtual environment and installing dependencies
        stage('Setting up Virtual Environment and installing dependencies') {
            steps {
                script {
                    echo 'Setting up Virtual Environment and installing dependencies' // Log message for virtual environment setup

                    // Shell commands to create a virtual environment, activate it, upgrade pip, and install dependencies
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        // Stage 3: Building and pushing a Docker image to Google Container Registry (GCR)
        stage('Building and Pushing Docker Image to GCR') {
            steps {
                // Using credentials to authenticate with GCP
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Building and Pushing Docker Image to GCR' // Log message for Docker build and push step

                        // Shell commands to authenticate with GCP, configure Docker, build the Docker image, and push it to GCR
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        docker build -t gcr.io/${GCP_PROJECT}/hotel-reservation-prediction:latest .
                        docker push gcr.io/${GCP_PROJECT}/hotel-reservation-prediction:latest
                        '''
                    }
                }
            }
        }
    }
}
