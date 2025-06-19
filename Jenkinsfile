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
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Sarvesh-Yadav-5201/MLOps_Project_1_Hotel_Reservation_prediction.git']])
            }
        }

}