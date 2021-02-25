pipeline {
    agent {
        dockerfile true
        label 'marshall'
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
            }
        }
    }
}
