## Docker documentation for Solar LSTNet
### Data Validation
#### **1. Prerequisite**
- You must configure the AWS credentials before start running the **data-validation** image. If you machine haven’t installed the awscli, install it.
- The AWS account must have the relevant permissions to interact with AWS services used in the Python script. Run this command and enter the AWS access key, also the AWS secret access key:
```
aws configure
```
- The path to AWS credentials stored at $HOME/.aws/credentials by default.

#### **2. Usage**
- The data-validation.py script need 3 parameters: weather_data (S3 path to the weather CSV file), building_data(S3 path to the building CSV file) and output_report (S3 path to save the validation report). So that you must declare it when running the data-validation image.
- Also, you need to mount the AWS credentials from your machine, which you configured before, to the docker container.
- Therefore, the command to run data-validation image is:
```
docker run -v $HOME/.aws/credentials:/root/.aws/credentials:ro -it data-validation --weather_data s3://trambk/solar-energy/raw_data/weather.csv --building_data s3://trambk/solar-energy/raw_data/Building_1.csv --output_report s3://trambk/solar-energy/raw_data/
```
#### **3. Result**
