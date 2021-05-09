output "ec2_public_ip" {
  value = module.worker.instance.public_ip
}