terraform {
  required_version = ">=0.12"
}

provider "aws" {
  region = "eu-west-1"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "worker-vpc"
  cidr = var.vpc_cidr_block

  azs             = [var.avail_zone]
  public_subnets  = [var.subnet_cidr_block]
  public_subnet_tags = { Name = "${var.env_prefix}-subnet-1" }

  tags = {
    Name = "${var.env_prefix}-vpc"
  }
}


module "worker" {
  source = "./modules/worker"
  image_name = "Deep Learning AMI (Amazon Linux 2) Version *"
  vpc_id = module.vpc.vpc_id
  env_prefix = var.env_prefix
  avail_zone = var.avail_zone
  instance_type = var.instance_type
  subnet_id = module.vpc.public_subnets[0]
  public_key_location = var.public_key_location
}

