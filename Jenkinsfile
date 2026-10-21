// CI/CD for road-damage-colombia, on a Jenkins running on the machine with the GPU.
// The dataset isn't in git: RDC_HOME points to a checkout that has data/yolo and runs/ (and the shared mlflow.db).
pipeline {
    agent any

    parameters {
        choice(name: 'WEIGHTS', choices: ['rtdetr_all', 'yolo_all', 'yolo_japan'],
               description: 'yolo_japan was trained on one country: pick it to watch the gate block it on Norway')
        booleanParam(name: 'BUILD_IMAGE', defaultValue: false, description: 'also build the API Docker image')
    }

    environment {
        PATH = "${env.HOME}/.local/bin:${env.HOME}/.local/opt/node/bin:${env.PATH}"
        WEIGHTS = "${params.WEIGHTS ?: 'rtdetr_all'}"
    }

    stages {
        stage('Install') {
            steps { sh 'uv sync --group mlops' }
        }

        stage('Lint and unit tests') {
            steps {
                sh 'uv run ruff check src scripts tests'
                sh 'mkdir -p reports && uv run pytest -q --junitxml=reports/pytest.xml'
            }
            post {
                always { junit 'reports/pytest.xml' }
            }
        }

        stage('Web build') {
            steps {
                dir('web') { sh 'npm ci && npm run build' }
            }
        }

        stage('Model quality gate (GPU)') {
            steps {
                sh '''
                    cd "${RDC_HOME:-.}"
                    uv run --group mlops scripts/quality_gate.py \
                        --weights "runs/$WEIGHTS/weights/best.pt" \
                        --tracking-uri "sqlite:///${RDC_HOME:-$PWD}/mlflow.db" \
                        --register
                '''
            }
        }

        stage('Docker image') {
            when { expression { params.BUILD_IMAGE } }
            steps { sh 'docker build -t road-damage-colombia-api:$BUILD_NUMBER .' }
        }
    }
}
