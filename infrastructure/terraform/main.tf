provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "ml_data" {
  bucket = var.s3_bucket_name
  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_instance" "ml_server" {
  ami           = var.ami_id
  instance_type = var.instance_type
  tags = {
    Name = "ML-Server"
  }
} 