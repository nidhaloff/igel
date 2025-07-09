# CloudFormation ML Infrastructure

This template provisions basic AWS infrastructure for ML workloads:
- S3 bucket for data/model storage
- EC2 instance for training/inference

## Usage

1. Go to the AWS CloudFormation console.
2. Upload `ml-infra.yaml` as a new stack.
3. Provide the required parameters (bucket name, AMI ID, instance type).
4. Launch the stack.

## Cost Optimization
- Use spot instances or smaller instance types for cost savings.
- Set S3 lifecycle policies for data retention.

## Monitoring
- Add CloudWatch alarms and logging as needed. 