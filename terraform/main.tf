/*
  Clyde — minimal single-instance AWS deployment.

  Provisions one Ubuntu 24.04 EC2 in the default VPC with a static
  Elastic IP, a security group permitting 22/80/443, and a cloud-init
  payload that:
    * installs Python + Caddy
    * clones the repo
    * writes /opt/clyde/.env from terraform vars
    * runs `python -m clyde.web` under systemd
    * fronts it with Caddy on :80 (or :443 with auto-LE if domain is set)

  Free-tier note: t2.micro / t3.micro qualify for 750 hrs/mo for 12
  months; t3.small (the default below) is more comfortable for the
  Monte-Carlo simulator but is NOT free-tier. Override via tfvars.
*/

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Latest Ubuntu 24.04 LTS (Canonical owner id).
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Use the account's default VPC + first available subnet so we don't
# spend a routing table / NAT just to ship a demo.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_key_pair" "clyde" {
  key_name   = "${var.name_prefix}-key"
  public_key = var.ssh_pub_key
}

resource "aws_security_group" "clyde" {
  name        = "${var.name_prefix}-sg"
  description = "Clyde web server (22/80/443)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-sg" }
}

resource "aws_instance" "clyde" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.clyde.key_name
  vpc_security_group_ids      = [aws_security_group.clyde.id]
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true

  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    repo_url       = var.repo_url
    repo_ref       = var.repo_ref
    domain         = var.domain
    cerebras_key   = var.cerebras_api_key
    openrouter_key = var.openrouter_api_key
    elevenlabs_key = var.elevenlabs_api_key
    clyde_provider = var.clyde_provider
    clyde_model    = var.clyde_model
  })
  # Re-running cloud-init on user_data change requires replacing the
  # instance, which is the right semantic for "redeploy with new config."
  user_data_replace_on_change = true

  root_block_device {
    volume_type = "gp3"
    volume_size = var.disk_gb
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  tags = {
    Name = "${var.name_prefix}-web"
    App  = "clyde"
  }
}

# Static public IP. NOTE: an unattached EIP is billed (~$0.005/h);
# destroy this when you tear the instance down or it stays charged.
resource "aws_eip" "clyde" {
  instance = aws_instance.clyde.id
  domain   = "vpc"
  tags     = { Name = "${var.name_prefix}-eip" }
}
