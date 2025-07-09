output "s3_bucket_arn" {
  value = aws_s3_bucket.ml_data.arn
}

output "ec2_instance_id" {
  value = aws_instance.ml_server.id
} 