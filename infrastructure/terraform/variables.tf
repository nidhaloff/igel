variable "region" {
  description = "AWS region"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name for the S3 bucket"
  type        = string
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
} 