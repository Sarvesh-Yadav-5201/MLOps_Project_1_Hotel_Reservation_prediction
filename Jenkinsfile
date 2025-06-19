//  only one step : Cloning actual repository to jenkins workspace

pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps {

                echo 'Cloning repository from GitHub to Jenkins workspace'
                // Cloning the repository
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Sarvesh-Yadav-5201/MLOps_Projects.git']])
            }
        }

    stages{
        stage('Setting up the Virtual Environment and installing dependencies'){     
            steps {

                echo 'Setting up the Virtual Environment and installing dependencies'
                sh '''
                python -m venv $VENV_DIR
                . ${VENV_DIR}/bin/activate

                pip install --upgrade pip
                pip install -e .

                '''
                
                
            }
        }
    }
}