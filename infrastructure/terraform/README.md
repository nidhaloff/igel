# Terraform ML Infrastructure

This template provisions basic AWS infrastructure for ML workloads:
- S3 bucket for data/model storage
- EC2 instance for training/inference

## Usage

1. Install [Terraform](https://www.terraform.io/).
2. Configure your AWS credentials.
3. Edit `variables.tf` to set your region, bucket name, AMI ID, and instance type.
4. Run:
   ```sh
   terraform init
   terraform apply
   ```
5. Resources will be created in your AWS account.

## Cost Optimization
- Use spot instances or smaller instance types for cost savings.
- Set S3 lifecycle policies for data retention.

## Monitoring
- Add CloudWatch alarms and logging as needed. 