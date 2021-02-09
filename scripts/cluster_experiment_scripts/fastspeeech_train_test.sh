#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

mkdir -p ${TMP}/data/
export DATA_DIR=${TMP}/data/

mkdir -p ${DATA_DIR}/models/
export MODEL_DIR=${DATA_DIR}/models/

#Download Data Set

# cd DATASET_DIR
# wget --header="Host: uoe-my.sharepoint.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.56" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://uoe-my.sharepoint.com/personal/s1661552_ed_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fs1661552%5Fed%5Fac%5Fuk%2FDocuments%2FMLP%20Data&originalPath=aHR0cHM6Ly91b2UtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwvczE2NjE1NTJfZWRfYWNfdWsvRWhwZnNqY0VMVkpIdm1rM1I2VGJuVXNCYjFCd01RRHp3QU5zbWJIYjM1TG1lQT9ydGltZT1EV0oyQW9ITTJFZw" --header="Cookie: MicrosoftApplicationsTelemetryDeviceId=1cbeb83d-9dc8-70d7-7001-8124cde52043; MicrosoftApplicationsTelemetryFirstLaunchTime=1612823528639; WSS_FullScreenMode=false; PowerPointWacDataCenter=UK4; WordWacDataCenter=GUK1; ExcelWacDataCenter=GUK1; WacDataCenter=GUK1; rtFa=Km8GouHKp0ZBjtFxUdsi0i6clYJq+CfYnh63kPBRL0smMEZDM0UzMEUtQTk5Ni00NERGLTgxQzYtQUJDOUI0NkNDODY3Ae/WZuikKkJVVtq83SzIDnIg71S8NydbpgbogKELKWFdQl70/t3Z1/vDVUMJORw6eDThh2RQ6lARtlvtGLbw7Byl+jG2i1K7+IvXbOXDazMPUHDVMX6qvj90rQrlMdFmkJ4vF/izV8M+mWl+mSNL8qzswGhpr7QroudvxMV7okUUe4F38Xil0dJfXt+mqV30Y4gCCZYFFhJhbG0ljJJOWAQACg/oixBERLd0Da8tTsMJAMOTRtip/j0ncRJHP4yKrdYg3QNXCCCWEGhHQjuHoW2iZ/2eq8s8KDnGSUDOa/El4szf0711BXpSu4a2EGMqbtavMGkENtvqYnD2bt8E30UAAAA=; CCSInfo=Mi8xMS8yMDIxIDExOjA0OjI5IFBNFwpeGYe5U4Du3oDG6iK36IG9Solp4kOLBIbtCRyD9odbDQURYXVnIPrc+jJZIDcc1qxo0kuVo3HAyp5gaqS0jO/QhaebpMj8przfbtka+Cl+2q57Apr+CJd+iVQLcfVc7xuDU2ycVyUuViJpaBQ8zcL0ZDXCf2gr0o7phviu/f6+y7sov/02v9G8fA1xFvxFNmhD9OIlKX3aa+bhb/trkvqnDp+Z0fpa4w5CyF2goDH6muZT+tygh4iEypi1vQlXndaSgVaQsm5tPRf9ssBevEFQjvNfJk4665hSuTdp/sShQlZ3rc8XIEIMLLYNlerFjcYaWRvXCsK4EzQW7rH8IhUAAAA=; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjgsMGguZnxtZW1iZXJzaGlwfDEwMDM3ZmZlYWQzMDliM2JAbGl2ZS5jb20sMCMuZnxtZW1iZXJzaGlwfHMxODQxMjE1QGVkLmFjLnVrLDEzMjUxNTY5MzQ5MDAwMDAwMCwxMzE3OTY5MjU1MDAwMDAwMDAsMTMyNTc3Mjg5NzQxNTgzOTc1LDE4OC43NC42NC4xMSwzLDBmYzNlMzBlLWE5OTYtNDRkZi04MWM2LWFiYzliNDZjYzg2NywsMmFkNTQzZTYtOTY4YS00NjBhLWE3MjgtNjA1ZTk1MTUwMTkxLDg4OThiM2RiLWI2M2QtNGZmMC1hMGNlLTE0NzM3OGY4MWQyMCwzZjAxZjkxMy1kZDFjLTRmM2YtOTk3OS0xZGRjNzEyZWRhYWMsLDAsMTMyNTczMDA1NzQxNDI3NzMwLDEzMjU3NTU2MTc0MTQyNzczMCwsLGV5SjRiWE5mWTJNaU9pSmJYQ0pEVURGY0lsMGlmUT09LDI2NTA0Njc3NDM5OTk5OTk5OTksMTMyNTcyOTE1NzAwMDAwMDAwLDUzNGUyNjUyLTcxOTgtNGJjNi05MDI0LTQ3NzIxN2ZlZTA2Zixoc2prL1p1cm9CMy82Y3dueWVMR0JKUm9icUlvaEpmdHNWeUZIUDFVcWpTMzZjVWY4ZGRwR1JsZmdBTUZkU1A5UVF4WUxvT3J4cVllMnVKdFAvV3BOWmhVQW4zT1VCTlZwUm1CNG9LSTQxZFBQRHd3cVdWSjcyOUk0RUxCRi9FcnQxQnFYWXBqSmRxSmFoM0U1UFJMbVEwdytWZnlRTlNZUkVlVmhHV1pGUW1mZ2dLYmpuRWVqajNkU1R1UmxEdGdBS0tEQ1pGTWcvbzJIMFZRSTBVaVFKZTlyNHFqSnNrV1VjMmQ0bGpWZkxoOFhlT05CVW4wMGttd1hPaGg0VVFxSWhMREJZd1JYQWhnS2pDUmZOb21yQktwOFByaUVvS0VSMG5zY1RGalJoSmVxYU95bGpTWXhzSzNCMGdtQmdBYUhsY1N6aXhBN2dTMXB6U3FQNGRBMVE9PTwvU1A+; odbn=1; cucg=1" --header="Connection: keep-alive" "https://uoe-my.sharepoint.com/personal/s1661552_ed_ac_uk/_layouts/15/download.aspx?UniqueId=5bfa827f%2D195c%2D4bf6%2D9bbc%2D8fc8d0af4810" -c -O 'LJSpeech.zip'
# unzip LJSpeech.zip -q

# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ../fastspeech2
python train.py --experiment-name 'train_test_exp'